"""
核心解析模型：从训练算力需求推导站点数与跨站带宽需求。

推演链条
---------
(1) 算力 → 功率
      P_facility = C / (T · MFU · η_facility)
(2) 功率 → 站点数
      N_power  = ceil(P_facility / P_site_max)
(3) 芯片供给 → 站点数下限（单站能装多少卡受供应链与建设节奏约束）
(4) 网络域 → 站点数下限（单个 scale-out 网络域能覆盖多少加速器）
      N_sites = max(N_power, N_supply, N_fabric)
(5) 站点数 + 同步策略 → 跨站链路带宽需求
      B_link = payload_bits / (sync_budget − t_prop)
(6) 与可得带宽比较 → 带宽缺口 / 最大可行站点数
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Optional

from params import (DemandScenario, AcceleratorScenario, SiteScenario, SyncStrategy,
                    BYTES_BF16, BITS_PER_BYTE)


@dataclass
class ScaleResult:
    # 规模
    peak_flops_needed: float
    facility_power_w: float
    n_dies: float
    n_sites_power: int
    n_sites_fabric: int
    n_sites: int
    power_per_site_w: float
    dies_per_site: float
    # 时间
    step_time_s: float
    n_steps: float
    # 通信
    payload_bits_per_sync: float
    sync_interval_s: float
    sync_budget_s: float
    prop_latency_s: float
    link_bps_required: float
    link_bps_available: float
    bandwidth_ratio: float          # required / available，>1 表示缺口
    feasible: bool
    # 元信息
    mfu: float
    eta_facility: float
    notes: str = ""

    def as_dict(self):
        return asdict(self)


def required_peak_flops(d: DemandScenario, mfu: float) -> float:
    return d.C_total_flop / (d.train_seconds * mfu)


def facility_power(d: DemandScenario, a: AcceleratorScenario, mfu: float) -> float:
    return required_peak_flops(d, mfu) / a.eta_facility_fp8


def ring_allreduce_per_link_bits(n_params: float, bits_per_param: float,
                                 n_sites: int, fragments: int = 1) -> float:
    """双向 ring all-reduce：每条相邻链路每次同步承载的比特数。

    对 M 个节点的 ring all-reduce，总量 2(M-1)/M·S 分 M 个 chunk 流转，
    稳态下每条链路单向承载约 S 比特（Epoch 的口径：相邻站点对交换约一个模型大小的载荷）。
    Streaming 分片把峰值摊平到 fragments 份。
    """
    if n_sites <= 1:
        return 0.0
    S_bits = n_params * bits_per_param
    return S_bits / max(fragments, 1)


def evaluate(d: DemandScenario,
             a: AcceleratorScenario,
             s: SiteScenario,
             strat: SyncStrategy,
             mfu: float = 0.35,
             n_sites_override: Optional[int] = None,
             chip_supply_limit_per_site: Optional[float] = None) -> ScaleResult:
    """对一组 (需求, 加速器, 站点, 策略) 求解站点数与带宽需求。"""
    peak = required_peak_flops(d, mfu)
    P = peak / a.eta_facility_fp8
    n_dies = peak / a.die_fp8_flops

    # (2) 功率口径
    n_sites_power = max(1, math.ceil(P / s.p_site_max_w))

    # (4) 网络域口径：单站 scale-out fabric 的加速器上限
    #     以 Google Virgo（单 fabric 最多 134,000 TPU）与 Meta FPF 的量级为参照，
    #     取 2030 年单站无阻塞 fabric 上限 2e5 die；超过则必须拆站或多 fabric。
    FABRIC_DIE_LIMIT = 2.0e5
    n_sites_fabric = max(1, math.ceil(n_dies / FABRIC_DIE_LIMIT))

    # (3) 供给口径（可选）
    n_sites_supply = 1
    if chip_supply_limit_per_site:
        n_sites_supply = max(1, math.ceil(n_dies / chip_supply_limit_per_site))

    n_sites = max(n_sites_power, n_sites_fabric, n_sites_supply)
    if n_sites_override:
        n_sites = n_sites_override
    if s.n_sites_available:
        n_sites = min(n_sites, s.n_sites_available)

    # (5) 通信
    t_step = d.step_time_s
    sync_interval = t_step * strat.H
    # eager/重叠策略：整个同步间隔都可用于传输；否则只给 sync_budget_frac
    budget = sync_interval * min(strat.sync_budget_frac, 1.0)
    t_prop = s.prop_latency_s_per_hop
    # ring 拓扑下传播时延为绕环一周
    t_prop_total = t_prop * n_sites if n_sites > 1 else 0.0

    payload_bits = ring_allreduce_per_link_bits(
        d.n_params_total, strat.bits_per_param, n_sites, strat.fragments)

    usable = budget - t_prop_total
    if usable <= 0:
        link_req = float("inf")
    else:
        link_req = payload_bits / usable

    link_avail = s.link_pbps * 1e15
    ratio = link_req / link_avail if link_avail > 0 else float("inf")

    return ScaleResult(
        peak_flops_needed=peak,
        facility_power_w=P,
        n_dies=n_dies,
        n_sites_power=n_sites_power,
        n_sites_fabric=n_sites_fabric,
        n_sites=n_sites,
        power_per_site_w=P / n_sites,
        dies_per_site=n_dies / n_sites,
        step_time_s=t_step,
        n_steps=d.n_steps,
        payload_bits_per_sync=payload_bits,
        sync_interval_s=sync_interval,
        sync_budget_s=budget,
        prop_latency_s=t_prop_total,
        link_bps_required=link_req,
        link_bps_available=link_avail,
        bandwidth_ratio=ratio,
        feasible=(ratio <= 1.0),
        mfu=mfu,
        eta_facility=a.eta_facility_fp8,
    )


def max_feasible_sites(d: DemandScenario, a: AcceleratorScenario, s: SiteScenario,
                       strat: SyncStrategy, mfu: float = 0.35,
                       cap: int = 500) -> int:
    """在给定链路带宽下，同步还能撑住的最大站点数。"""
    best = 0
    for n in range(1, cap + 1):
        r = evaluate(d, a, s, strat, mfu, n_sites_override=n)
        if r.feasible:
            best = n
        else:
            break
    return best


# ---------------------------------------------------------------- 故障率模型
def goodput(n_dies: float, mtbf_die_years: float, recovery_s: float,
            checkpoint_interval_s: float, elastic: bool = False,
            quorum_frac: float = 1.0) -> float:
    """同步训练的 goodput（有效训练时间占比）。

    同步数据并行：任一芯片故障即全局停顿，损失 = 恢复时间 + 回放的半个 checkpoint 区间。
    弹性/quorum 方案：只要在线比例 ≥ quorum_frac 就继续，损失退化为被剔除份额。

    标定：Llama 3 405B，16K GPU，54 天 419 次非预期中断 ≈ 3.09 h/次
          → 单卡 MTBI ≈ 16384 × 3.09 h ≈ 5.8 年（arXiv:2407.21783 Table 5）
    """
    mtbf_s = mtbf_die_years * 365.25 * 86400
    rate = n_dies / mtbf_s                      # 全局故障率 (1/s)
    if not elastic:
        loss_per_fault = recovery_s + checkpoint_interval_s / 2.0
        return 1.0 / (1.0 + rate * loss_per_fault)
    # 弹性：单点故障只损失该 learner island 的产出，直到被替换
    island_frac = 1.0 - quorum_frac + 1e-9
    loss_per_fault = recovery_s * max(island_frac, 1.0 / max(n_dies, 1))
    return 1.0 / (1.0 + rate * loss_per_fault)
