"""
Roofline + 通信时间线仿真（per-die 口径）

对一个训练 step 建模 5D 并行（TP / CP / PP / EP / DP）在三级网络层次上的
**单 die** 通信量与暴露时间，输出 MFU 与时间分解。

自洽性
-------
仿真不独立指定 microbatch 数，而是由全局批量反推，避免配置与规模脱节：
    dies_per_replica = tp · cp · pp · ep
    n_replicas       = N_total / dies_per_replica
    tokens_per_replica_step = global_batch / n_replicas
    m (microbatch 数) = tokens_per_replica_step / (b · s)
这样 "关键批量大小限制了能并行多宽" 这件事会自然显现为 MFU 的塌陷。

网络层次
---------
  L0  scale-up 域内   NVLink / 灵衢 UnifiedBus     单 die 双向 TB/s 级
  L1  站内 scale-out  RoCE / IB                    单 die 单向 百 Gb/s 级
  L2  跨站 WAN        相干 DWDM                     站点对 Pbps 级（全站共享）

单 die 通信量（记 Ls = L/pp，tok_die = b·s/(ep·cp)，h = d_model）
  TP   8 · Ls · tok_die · h · (tp−1)/tp · 2B        每 microbatch，scale-up
       （fwd 2 次 + bwd 2 次 AllReduce，AllReduce 体积 = 2m(t−1)/t）
  CP   4 · Ls · tok_die · h · kv · (cp−1)/cp · 2B   每 microbatch，scale-up
  EP   4 · Ls · tok_die · topk · h · (ep−1)/ep · 2B 每 microbatch，
       ep·tp ≤ scale-up 域则走 L0，否则走 L1
  PP   2 · tok_die · h · 2B                         每 microbatch，L1（或 L2）
  DP   2 · (P/(tp·pp·ep)) · (dp−1)/dp · 2B          每 step，L1
  WAN  P · bits_per_param / fragments               每 H 个 step，L2（站点对共享）

暴露系数依据
-------------
  TP / CP 完全暴露：Meta Llama 3 ISCA'25 明确 "completely exposed on critical path"
  EP      站内 DualPipe 可掩盖大部分（DeepSeek-V3）；跨节点掩盖能力下降
          （HybridEP arXiv:2510.19470：低带宽下 A2A 可占训练时间 90%+）
  PP      主要暴露在 warm-up/cool-down；气泡单独建模
  DP      与 fwd/bwd 重叠，只有超出计算窗口的部分暴露
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict

BYTES = 2  # BF16

EXPOSURE = {
    "tp": 1.00,
    "cp": 1.00,
    "ep_intra_scaleup": 0.15,
    "ep_cross_node": 0.55,
    "pp": 0.20,
}
TP_OPS_PER_LAYER = 8      # fwd 2 + bwd 2 次 AllReduce，每次体积 2m(t−1)/t
CP_OPS_PER_LAYER = 4
EP_OPS_PER_LAYER = 4      # dispatch + combine，fwd + bwd


@dataclass
class ParallelConfig:
    tp: int = 8
    cp: int = 1
    pp: int = 16
    ep: int = 32
    n_sites: int = 1
    micro_batch: int = 1
    dualpipe: bool = True
    zero_stage: int = 1

    @property
    def dies_per_replica(self) -> int:
        return self.tp * self.cp * self.pp * self.ep


@dataclass
class NetworkHierarchy:
    scaleup_bw_bps: float
    scaleup_domain: int
    scaleout_bw_bps: float
    wan_bw_bps: float
    wan_latency_s: float
    scaleup_latency_s: float = 2e-7
    scaleout_latency_s: float = 5e-6


@dataclass
class StepBreakdown:
    n_total_dies: float
    dies_per_replica: int
    n_replicas: float
    dp_degree: float
    tokens_per_die_step: float
    micro_batches: float
    t_compute: float
    t_tp: float
    t_cp: float
    t_ep: float
    t_pp: float
    t_dp_intra: float
    t_dp_inter: float
    t_bubble: float
    t_exposed: float
    t_step: float
    mfu: float
    params_per_die: float
    hbm_pressure_gb: float
    bytes_wan_per_sync: float
    wan_link_util: float
    bound_by: str
    scaleup_fits: bool

    def as_dict(self):
        return asdict(self)


def simulate_step(*, n_layers: int, d_model: int, seq_len: int,
                  n_params_total: float, n_params_active: float,
                  global_batch_tokens: float,
                  topk: int, kv_frac: float,
                  n_total_dies: float,
                  cfg: ParallelConfig, net: NetworkHierarchy,
                  die_fp8_flops: float, hbm_gb_per_die: float,
                  sync_every: int = 1, wan_bits_per_param: float = 16,
                  wan_fragments: int = 1,
                  pp_crosses_wan: bool = False) -> StepBreakdown:
    L, h, s_len, b = n_layers, d_model, seq_len, cfg.micro_batch
    Ls = L / cfg.pp

    dpr = cfg.dies_per_replica
    n_replicas = max(n_total_dies / dpr, 1.0)
    dp = n_replicas                       # 含跨站的总数据并行度
    dp_intra = max(dp / max(cfg.n_sites, 1), 1.0)

    tokens_per_replica_step = global_batch_tokens / n_replicas
    m = max(tokens_per_replica_step / (b * s_len), 1.0)
    tok_die = b * s_len / (cfg.ep * cfg.cp)

    flops_per_token = 6.0 * n_params_active + 12.0 * L * h * s_len
    tokens_per_die_step = tokens_per_replica_step / dpr * (cfg.ep * cfg.cp)

    # ---- 计算 ----
    t_compute = (tokens_per_replica_step * flops_per_token / dpr) / die_fp8_flops

    # ---- TP ----
    by_tp = TP_OPS_PER_LAYER * Ls * tok_die * h * (cfg.tp - 1) / max(cfg.tp, 1) * BYTES
    t_tp = (by_tp * 8 / net.scaleup_bw_bps + TP_OPS_PER_LAYER * Ls * net.scaleup_latency_s) * m

    # ---- CP ----
    if cfg.cp > 1:
        by_cp = CP_OPS_PER_LAYER * Ls * tok_die * h * kv_frac * (cfg.cp - 1) / cfg.cp * BYTES
    else:
        by_cp = 0.0
    t_cp = by_cp * 8 / net.scaleup_bw_bps * m

    # ---- EP ----
    if cfg.ep > 1:
        by_ep = EP_OPS_PER_LAYER * Ls * tok_die * topk * h * (cfg.ep - 1) / cfg.ep * BYTES
        intra = (cfg.ep * cfg.tp) <= net.scaleup_domain
        bw_ep = net.scaleup_bw_bps if intra else net.scaleout_bw_bps
        exp_ep = EXPOSURE["ep_intra_scaleup"] if intra else EXPOSURE["ep_cross_node"]
        t_ep = by_ep * 8 / bw_ep * m
    else:
        by_ep, t_ep, exp_ep, intra = 0.0, 0.0, 0.0, True

    # ---- PP ----
    if cfg.pp > 1:
        by_pp = 2 * tok_die * h * BYTES
        bw_pp = net.wan_bw_bps if pp_crosses_wan else net.scaleout_bw_bps
        t_pp = by_pp * 8 / bw_pp * m
        bubble_frac = (cfg.pp - 1) / (m + cfg.pp - 1)
        if cfg.dualpipe:
            bubble_frac *= 0.5
        t_bubble = t_compute * bubble_frac
    else:
        by_pp, t_pp, t_bubble = 0.0, 0.0, 0.0

    # ---- DP（站内）----
    params_per_die = n_params_total / (cfg.tp * cfg.pp * cfg.ep)
    zero_mult = 3.0 if cfg.zero_stage >= 3 else 2.0
    if dp_intra > 1:
        by_dp = zero_mult * params_per_die * (dp_intra - 1) / dp_intra * BYTES
        t_dp_intra = by_dp * 8 / net.scaleout_bw_bps
    else:
        t_dp_intra = 0.0

    # ---- 跨站 ----
    if cfg.n_sites > 1:
        payload_bits = n_params_total * wan_bits_per_param / max(wan_fragments, 1)
        t_ser = payload_bits / net.wan_bw_bps
        t_prop = net.wan_latency_s * cfg.n_sites
        t_dp_inter = (t_ser + t_prop) / max(sync_every, 1)
        bytes_wan = payload_bits / 8
    else:
        t_dp_inter, bytes_wan, t_ser = 0.0, 0.0, 0.0

    # ---- 合成 ----
    t_exposed_comm = (EXPOSURE["tp"] * t_tp + EXPOSURE["cp"] * t_cp
                      + exp_ep * t_ep + EXPOSURE["pp"] * t_pp)
    t_dp_exposed = max(0.0, (t_dp_intra + t_dp_inter) - t_compute)
    t_exposed = t_exposed_comm + t_dp_exposed
    t_step = t_compute + t_exposed + t_bubble
    mfu = t_compute / t_step if t_step > 0 else 0.0

    parts = {"计算": t_compute, "TP": EXPOSURE["tp"] * t_tp, "CP": EXPOSURE["cp"] * t_cp,
             "EP": exp_ep * t_ep, "PP": EXPOSURE["pp"] * t_pp,
             "DP": t_dp_exposed, "气泡": t_bubble}
    bound = max(parts, key=parts.get)

    # HBM 压力：参数 + 优化器状态(FP32 m,v + master) ≈ 参数的 8 倍字节
    hbm_gb = params_per_die * (2 + 12) / 1e9

    return StepBreakdown(
        n_total_dies=n_total_dies, dies_per_replica=dpr, n_replicas=n_replicas,
        dp_degree=dp, tokens_per_die_step=tokens_per_die_step, micro_batches=m,
        t_compute=t_compute, t_tp=t_tp, t_cp=t_cp, t_ep=t_ep, t_pp=t_pp,
        t_dp_intra=t_dp_intra, t_dp_inter=t_dp_inter, t_bubble=t_bubble,
        t_exposed=t_exposed, t_step=t_step, mfu=mfu,
        params_per_die=params_per_die, hbm_pressure_gb=hbm_gb,
        bytes_wan_per_sync=bytes_wan,
        wan_link_util=(t_ser / (t_step * max(sync_every, 1)) if cfg.n_sites > 1 and t_step > 0 else 0.0),
        bound_by=bound, scaleup_fits=(cfg.tp * cfg.cp * cfg.ep <= net.scaleup_domain),
    )
