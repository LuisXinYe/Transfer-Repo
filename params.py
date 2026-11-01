"""
2030 多数据中心联合训练 —— 参数库

所有参数附来源标注。单位统一：
  FLOP, FLOP/s, W, bit, byte, s, m
命名约定：
  *_FP8 表示稠密 FP8 口径；本模型统一用稠密 FP8 作为训练算力基准单位。
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------- 常量
S_PER_DAY = 86400.0
C_FIBER_US_PER_KM = 5.0          # 光在标准单模光纤中的单向时延 μs/km (CommScope WP-111432-EN)
C_HCF_US_PER_KM = 3.44           # 空芯光纤，群折射率≈1.003 (arXiv:2606.23511)
BYTES_BF16 = 2
BITS_PER_BYTE = 8


# ---------------------------------------------------------------- 需求侧
@dataclass
class DemandScenario:
    """2030 年单次前沿训练任务的规模假设。"""
    name: str
    # 训练算力
    C_total_flop: float            # 一次训练的总 FLOP
    train_days: float              # wall-clock 训练天数
    # 模型形态
    n_params_total: float          # 总参数量
    n_params_active: float         # 每 token 激活参数量
    n_tokens: float                # 训练 token 数
    # 训练动力学
    batch_tokens: float            # 全局批量（token）
    # 结构（用于通信时间线仿真）
    n_layers: int
    d_model: int
    seq_len: int
    source: str = ""

    @property
    def train_seconds(self) -> float:
        return self.train_days * S_PER_DAY

    @property
    def flops_per_token(self) -> float:
        """前向+反向，约 6×激活参数（不含 attention 的 O(s²) 项）。"""
        return 6.0 * self.n_params_active

    @property
    def n_steps(self) -> float:
        return self.n_tokens / self.batch_tokens

    @property
    def throughput_eff(self) -> float:
        """达成该任务所需的有效（而非峰值）算力 FLOP/s。"""
        return self.C_total_flop / self.train_seconds

    @property
    def step_time_s(self) -> float:
        return self.batch_tokens * self.flops_per_token / self.throughput_eff

    @property
    def grad_bytes(self) -> float:
        """一次全量梯度同步的载荷（BF16）。"""
        return self.n_params_total * BYTES_BF16


# ---------------------------------------------------------------- 供给侧：加速器
@dataclass
class AcceleratorScenario:
    """2030 年量产加速器的系统级能效。

    eta_facility_fp8: 设施级峰值能效 (FLOP/s per W)，已折算：
      die 级算力 → 封装/整卡 → 整机（CPU/NIC/内存）→ PUE
    即  设施总功率 = 峰值算力 / eta_facility_fp8
    """
    name: str
    die_fp8_flops: float           # 单 die 稠密 FP8 峰值 FLOP/s
    die_power_w: float             # 单 die 功耗（含其份额的 HBM）
    gpu_fraction_of_facility: float  # 加速器功耗 / 设施总功率
    hbm_gb_per_die: float
    hbm_bw_tbs: float              # 单 die HBM 带宽 TB/s
    scaleup_bw_tbs: float          # 单 die scale-up 双向带宽 TB/s
    scaleout_bw_gbs: float         # 单 die scale-out（RoCE/IB）单向 Gb/s
    scaleup_domain: int            # scale-up 域内 die 数
    source: str = ""

    @property
    def eta_facility_fp8(self) -> float:
        return (self.die_fp8_flops / self.die_power_w) * self.gpu_fraction_of_facility

    @property
    def die_facility_power_w(self) -> float:
        return self.die_power_w / self.gpu_fraction_of_facility


# ---------------------------------------------------------------- 供给侧：站点
@dataclass
class SiteScenario:
    name: str
    p_site_max_w: float            # 单站可用总功率上限（含 PUE）
    n_sites_available: Optional[int] = None   # 可用于同一任务的站点数上限
    inter_site_km: float = 1000.0  # 相邻站点光纤路由距离
    link_pbps: float = 1.0         # 相邻站点对之间单向可用带宽 (Pbps)
    hollow_core: bool = False
    source: str = ""

    @property
    def prop_latency_s_per_hop(self) -> float:
        us_per_km = C_HCF_US_PER_KM if self.hollow_core else C_FIBER_US_PER_KM
        return self.inter_site_km * us_per_km * 1e-6


# ---------------------------------------------------------------- 同步策略
@dataclass
class SyncStrategy:
    """跨站同步策略。"""
    name: str
    H: int                         # 同步间隔（内层步数），1 = 同步数据并行
    bits_per_param: float          # 跨站传输每参数比特数（BF16=16, 4bit量化=4, 1bit符号=1）
    fragments: int = 1             # Streaming 分片数：峰值带宽 ÷ fragments
    overlap_steps: int = 0         # 可与计算重叠的外层步数（eager update）
    sync_budget_frac: float = 0.05 # 允许同步占 wall-clock 的比例
    quality_penalty: str = ""      # 已知的精度代价
    evidence: str = ""             # 验证规模


# ================================================================
# 实例化：需求侧场景
# ================================================================
# 锚点：Epoch AI 估 Grok 4 ≈ 5e26 FLOP；2026 年前沿单次训练 ~1e27 FLOP。
# 训练算力历史增速 4–5x/yr（Epoch，2010–2024），2018 后约 4.2x/yr。
# 从 2026-10 到 2030-06 约 3.7 年。
_ANCHOR_2026 = 1.0e27
_YEARS = 3.7

DEMAND = {
    "conservative": DemandScenario(
        name="保守（3x/yr）",
        C_total_flop=_ANCHOR_2026 * 3.0 ** _YEARS,        # ≈ 4.4e28
        train_days=120,
        n_params_total=20e12, n_params_active=1.2e12, n_tokens=600e12,
        batch_tokens=120e6,
        n_layers=120, d_model=16384, seq_len=32768,
        source="Epoch AI 4-5x/yr 趋势下限；锚点 Grok4≈5e26 FLOP",
    ),
    "central": DemandScenario(
        name="中性（4x/yr）",
        C_total_flop=_ANCHOR_2026 * 4.0 ** _YEARS,        # ≈ 1.5e29
        train_days=100,
        n_params_total=72e12, n_params_active=4e12, n_tokens=2000e12,
        batch_tokens=200e6,
        n_layers=160, d_model=24576, seq_len=32768,
        source="Epoch AI 中位趋势；模型形态沿用 Epoch 10GW 推演 (72T/4T/2000T)",
    ),
    "aggressive": DemandScenario(
        name="激进（5x/yr）",
        C_total_flop=_ANCHOR_2026 * 5.0 ** _YEARS,        # ≈ 3.9e29
        train_days=100,
        n_params_total=150e12, n_params_active=8e12, n_tokens=3000e12,
        batch_tokens=400e6,
        n_layers=200, d_model=32768, seq_len=32768,
        source="Epoch AI 趋势上限，接近其 2e29 FLOP 上界",
    ),
}

# Epoch AI 自身推演（用于交叉校验本模型）
DEMAND["epoch_10gw"] = DemandScenario(
    name="Epoch 10GW 推演（校验用）",
    C_total_flop=5e28, train_days=100,
    n_params_total=72e12, n_params_active=4e12, n_tokens=2000e12,
    batch_tokens=1.1e9,
    n_layers=160, d_model=24576, seq_len=32768,
    source="epoch.ai/blog/could-decentralized-training-solve-ais-power-problem",
)


# ================================================================
# 实例化：加速器场景
# ================================================================
# 参照点（2026）：
#   GB200 die: 5 PFLOPS FP4 dense/die → 2.5 PFLOPS FP8，600 W/die
#   Rubin (2026H2): 50 PFLOPS FP4/package(2 die) → 25 PF FP4/die → 12.5 PF FP8/die,
#                   1800 W/package → 900 W/die, 288GB HBM4/package, 13 TB/s
#   Rubin Ultra (2027H2): 100 PF FP4/package(4 die), 3600 W/package → 900 W/die,
#                   1TB HBM4E/package, 32 TB/s
#   (tomshardware NVIDIA roadmap; NVIDIA GTC)
#   设施占比：Epoch 校准 —— 132 kW 机架标称 → 200 kW 装机（含 PUE 1.35），
#            GPU 占机架 ~65% → GPU/设施 ≈ 0.43
ACCEL = {
    "us_2030": AcceleratorScenario(
        name="美国路径 2030（Feynman 后继）",
        die_fp8_flops=30e15,       # Rubin die 12.5 PF FP8 × ~2.4（两代，先进制程+封装）
        die_power_w=1100,
        gpu_fraction_of_facility=0.46,   # PUE 1.15–1.25 + 更高占比的加速器功耗
        hbm_gb_per_die=400, hbm_bw_tbs=20, scaleup_bw_tbs=3.6, scaleout_bw_gbs=800,
        scaleup_domain=576,
        source="由 Rubin/Rubin Ultra 公开规格（tomshardware）外推两代；设施占比按 Epoch 校准",
    ),
    "us_2030_low": AcceleratorScenario(
        name="美国路径 2030（悲观）",
        die_fp8_flops=20e15, die_power_w=1200, gpu_fraction_of_facility=0.43,
        hbm_gb_per_die=300, hbm_bw_tbs=15, scaleup_bw_tbs=3.6, scaleout_bw_gbs=800,
        scaleup_domain=288,
        source="制程与 HBM 推进放缓情形",
    ),
    "cn_2030": AcceleratorScenario(
        name="中国路径 2030（昇腾 980 后继）",
        # 昇腾 970 (2028Q4): 4 PFLOPS FP8 / 8 PFLOPS FP4（2025 路线图口径）
        #   Epoch 引 2026 更新版：970 = 14 PF FP4 (2028)、980 = 28 PF FP4 (2029)
        #   取 980 = 14 PF FP8，2030 后继小幅提升到 18 PF FP8
        die_fp8_flops=18e15,
        die_power_w=1500,          # 由 Atlas 350(950PR) 600W @0.78PF FP8 外推
        gpu_fraction_of_facility=0.44,   # PUE 更优(1.1–1.2)，但单板外围占比更高
        hbm_gb_per_die=288, hbm_bw_tbs=14.4, scaleup_bw_tbs=4.0, scaleout_bw_gbs=400,
        scaleup_domain=4096,       # 灵衢/UnifiedBus，Atlas 960 超节点 4096 卡统一编址
        source="华为 2025-09 全联接大会路线图 + Epoch 2026 更新版；功耗由 Atlas 350 实测 600W 外推",
    ),
    "cn_2030_low": AcceleratorScenario(
        name="中国路径 2030（受制程/HBM 约束）",
        die_fp8_flops=8e15, die_power_w=1500, gpu_fraction_of_facility=0.42,
        hbm_gb_per_die=192, hbm_bw_tbs=8, scaleup_bw_tbs=2.2, scaleout_bw_gbs=400,
        scaleup_domain=4096,
        source="DUV 多重曝光良率与国产 HBM 产能受限情形",
    ),
    # 校验用
    "gb200": AcceleratorScenario(
        name="GB200（2025，校验用）",
        die_fp8_flops=2.5e15, die_power_w=600, gpu_fraction_of_facility=0.43,
        hbm_gb_per_die=96, hbm_bw_tbs=4, scaleup_bw_tbs=1.8, scaleout_bw_gbs=400,
        scaleup_domain=72,
        source="NVL72 132kW 标称 → 200kW 装机（Epoch 校准）",
    ),
}


# ================================================================
# 实例化：站点场景
# ================================================================
SITES = {
    "us_1gw": SiteScenario("美国 1 GW 园区", 1.0e9, inter_site_km=1000, link_pbps=1.0,
                           source="Stargate Abilene 1.2GW / Prometheus ~1GW 量级"),
    "us_2gw": SiteScenario("美国 2 GW 园区", 2.0e9, inter_site_km=1000, link_pbps=3.0,
                           source="Stargate Shackelford ~2GW (Epoch 核查口径)"),
    "us_5gw": SiteScenario("美国 5 GW 园区", 5.0e9, inter_site_km=1200, link_pbps=6.0,
                           source="Meta Hyperion 5GW 规划，首期 2GW 目标 2030"),
    "cn_0p5gw": SiteScenario("中国 0.5 GW 园区", 0.5e9, inter_site_km=1200, link_pbps=0.3,
                             source="当前最大单体：世纪互联乌兰察布 IT 221MW / 华为和林格尔 242MW"),
    "cn_1gw": SiteScenario("中国 1 GW 园区", 1.0e9, inter_site_km=1200, link_pbps=0.6,
                           source="世纪互联乌兰察布总规划 1.3GW IT；DeepSeek 拟建 1GW"),
    "cn_2gw": SiteScenario("中国 2 GW 园区", 2.0e9, inter_site_km=1500, link_pbps=1.2,
                           source="远景乌兰察布星河基地规划电力容量 2GW（2026-08 投产）"),
}


# ================================================================
# 实例化：同步策略
# ================================================================
STRATEGIES = {
    "sync_dp": SyncStrategy(
        "同步数据并行", H=1, bits_per_param=16, fragments=1,
        quality_penalty="无（基准）",
        evidence="前沿规模标准做法；Epoch 10GW 推演采用",
    ),
    "hier_ar": SyncStrategy(
        "分层 All-Reduce（站内 RS→站间 AR→站内 AG）", H=1, bits_per_param=16, fragments=1,
        quality_penalty="无（数学等价）",
        evidence="NVIDIA NeMo：Nemotron-4 340B / 3072 GPU / 1000km / RTT 21ms，scaling efficiency >96%",
    ),
    "diloco_h30": SyncStrategy(
        "DiLoCo H=30", H=30, bits_per_param=16, fragments=1,
        quality_penalty="2.4B 处 M=2 已反超 DP；M=8 时 1.3B 处 +1.3% loss",
        evidence="arXiv:2503.09799，全超参扫到 2.4B，4B/10B 外推验证",
    ),
    "diloco_h100_q4": SyncStrategy(
        "Streaming DiLoCo H=100 + 4bit", H=100, bits_per_param=4, fragments=8,
        quality_penalty="量化带来的 loss gap <0.12%",
        evidence="arXiv:2501.18512，35M–4B；1B/Dolma 250B token",
    ),
    "eager_h100_q4": SyncStrategy(
        "Eager DiLoCo H=100 + 4bit + 完全重叠", H=100, bits_per_param=4, fragments=8,
        overlap_steps=1, sync_budget_frac=1.0,
        quality_penalty="1B/250B token：eval loss 2.46 vs DP 2.45",
        evidence="arXiv:2502.12996；100B 仿真 0.4 Gbit/s vs DP 471.5",
    ),
    "decoupled_diloco": SyncStrategy(
        "Decoupled DiLoCo（quorum K=1, P=24, H=24）", H=24, bits_per_param=4, fragments=24,
        overlap_steps=2, sync_budget_frac=1.0,
        quality_penalty="2B–9B dense + MoE 与 DP 打平",
        evidence="arXiv:2604.21428；1.2M 芯片仿真 goodput 88% vs DP 27%",
    ),
}


def dump_params():
    import json
    out = {
        "demand": {k: asdict(v) for k, v in DEMAND.items()},
        "accelerator": {k: {**asdict(v), "eta_facility_fp8": v.eta_facility_fp8}
                        for k, v in ACCEL.items()},
        "sites": {k: asdict(v) for k, v in SITES.items()},
        "strategies": {k: asdict(v) for k, v in STRATEGIES.items()},
    }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dump_params(), ensure_ascii=False, indent=2, default=str))
