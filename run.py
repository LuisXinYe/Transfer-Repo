"""
场景扫描 + 通信时间线仿真 + 蒙特卡洛敏感性分析

输出：
  data/validation.csv        与 Epoch AI 10GW 推演的交叉校验
  data/scenarios.csv         主场景表（中美 × 需求 × 站点规模 × 同步策略）
  data/bandwidth_sweep.csv   同步策略 × 站点数 → 带宽需求
  data/timeline.csv          roofline/通信时间线仿真
  data/montecarlo.csv        蒙特卡洛样本
  data/sensitivity.csv       敏感性排序
  data/summary.json          供交互页面使用
"""
from __future__ import annotations
import csv, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import DEMAND, ACCEL, SITES, STRATEGIES, DemandScenario
from core import evaluate, max_feasible_sites, required_peak_flops, goodput
from timeline import ParallelConfig, NetworkHierarchy, simulate_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

PB = 1e15
GW = 1e9


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(os.path.join(DATA, path), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  wrote data/{path}  ({len(rows)} rows)")


# ============================================================ 1. 校验
def validation():
    """用 Epoch AI 的 10 GW 推演反算，确认本模型的量纲与系数自洽。"""
    d = DEMAND["epoch_10gw"]
    a = ACCEL["gb200"]
    rows = []

    peak = required_peak_flops(d, mfu=0.30)
    P = peak / a.eta_facility_fp8
    rows.append(dict(项="峰值算力 FLOP/s", 本模型=f"{peak:.3g}", Epoch原文="2e22", 单位="FLOP/s"))
    rows.append(dict(项="设施总功率", 本模型=f"{P/GW:.2f}", Epoch原文="10", 单位="GW"))
    rows.append(dict(项="加速器封装数（2 die/封装）", 本模型=f"{peak/a.die_fp8_flops/2/1e6:.2f}",
                     Epoch原文="3.7", 单位="百万 GPU"))
    rows.append(dict(项="step 时间", 本模型=f"{d.step_time_s:.2f}", Epoch原文="<4.8", 单位="s"))
    rows.append(dict(项="梯度载荷", 本模型=f"{d.grad_bytes*8/PB:.2f}", Epoch原文="1.12", 单位="Pbit"))
    rows.append(dict(项="梯度更新次数", 本模型=f"{d.n_steps/1e6:.2f}", Epoch原文="1.8", 单位="百万"))

    s = SITES["us_1gw"]
    s2 = type(s)(name="Epoch ring", p_site_max_w=10e9/23, inter_site_km=4800/23,
                 link_pbps=6.0)
    r = evaluate(d, a, s2, STRATEGIES["sync_dp"], mfu=0.30, n_sites_override=23)
    rows.append(dict(项="环路传播时延", 本模型=f"{r.prop_latency_s*1000:.0f}", Epoch原文="24", 单位="ms"))
    rows.append(dict(项="序列化时间 @6Pbps", 本模型=f"{r.payload_bits_per_sync/6e15*1000:.0f}",
                     Epoch原文="190", 单位="ms"))
    rows.append(dict(项="所需链路带宽", 本模型=f"{r.link_bps_required/PB:.2f}", Epoch原文="6", 单位="Pbps"))
    write_csv("validation.csv", rows)
    return rows


# ============================================================ 2. 主场景表
PATHS = [
    # (路径名, 需求key, 加速器key, [站点key], mfu)
    ("美国", "conservative", "us_2030", ["us_1gw", "us_2gw", "us_5gw"], 0.38),
    ("美国", "central",      "us_2030", ["us_1gw", "us_2gw", "us_5gw"], 0.35),
    ("美国", "aggressive",   "us_2030", ["us_2gw", "us_5gw"],           0.33),
    ("美国", "central",      "us_2030_low", ["us_1gw", "us_2gw", "us_5gw"], 0.35),
    ("中国", "conservative", "cn_2030", ["cn_0p5gw", "cn_1gw", "cn_2gw"], 0.33),
    ("中国", "central",      "cn_2030", ["cn_0p5gw", "cn_1gw", "cn_2gw"], 0.30),
    ("中国", "aggressive",   "cn_2030", ["cn_1gw", "cn_2gw"],             0.28),
    ("中国", "central",      "cn_2030_low", ["cn_0p5gw", "cn_1gw", "cn_2gw"], 0.30),
]


def scenarios():
    rows = []
    for path, dk, ak, sks, mfu in PATHS:
        d, a = DEMAND[dk], ACCEL[ak]
        for sk in sks:
            s = SITES[sk]
            for stk, strat in STRATEGIES.items():
                r = evaluate(d, a, s, strat, mfu=mfu)
                nmax = max_feasible_sites(d, a, s, strat, mfu=mfu, cap=300)
                rows.append(dict(
                    路径=path, 需求场景=d.name, 加速器=a.name, 站点规模=s.name, 同步策略=strat.name,
                    训练算力_FLOP=f"{d.C_total_flop:.3g}",
                    设施能效_TFLOPs每W=round(a.eta_facility_fp8 / 1e12, 2),
                    MFU=mfu,
                    总功率_GW=round(r.facility_power_w / GW, 2),
                    加速器数_万=round(r.n_dies / 1e4, 1),
                    站点数=r.n_sites,
                    单站功率_MW=round(r.power_per_site_w / 1e6, 0),
                    step时间_s=round(r.step_time_s, 2),
                    同步间隔_s=round(r.sync_interval_s, 1),
                    单次载荷_Pbit=round(r.payload_bits_per_sync / PB, 3),
                    所需链路_Pbps=(round(r.link_bps_required / PB, 4)
                                   if math.isfinite(r.link_bps_required) else "∞"),
                    可得链路_Pbps=round(r.link_bps_available / PB, 2),
                    带宽缺口倍数=(round(r.bandwidth_ratio, 2)
                                   if math.isfinite(r.bandwidth_ratio) else "∞"),
                    可行=("是" if r.feasible else "否"),
                    该策略最大可行站点数=nmax,
                ))
    write_csv("scenarios.csv", rows)
    return rows


# ============================================================ 3. 带宽扫描
def bandwidth_sweep():
    rows = []
    for path, dk, ak, mfu in [("美国", "central", "us_2030", 0.35),
                              ("中国", "central", "cn_2030", 0.30)]:
        d, a = DEMAND[dk], ACCEL[ak]
        s = SITES["us_2gw"] if path == "美国" else SITES["cn_1gw"]
        for stk, strat in STRATEGIES.items():
            for n in [2, 4, 8, 12, 16, 24, 32, 48, 64, 100]:
                r = evaluate(d, a, s, strat, mfu=mfu, n_sites_override=n)
                rows.append(dict(
                    路径=path, 同步策略=strat.name, 站点数=n,
                    所需链路_Pbps=(round(r.link_bps_required / PB, 5)
                                   if math.isfinite(r.link_bps_required) else None),
                    传播时延_ms=round(r.prop_latency_s * 1000, 1),
                    同步预算_s=round(r.sync_budget_s, 2),
                    可行=("是" if r.feasible else "否"),
                ))
    write_csv("bandwidth_sweep.csv", rows)
    return rows


# ============================================================ 4. 时间线仿真
TIMELINE_CASES = [
    # (标签, 需求key, 加速器key, mfu_for_sizing, 并行配置, 同步策略, 全局批量倍率)
    ("美国·单站基线（576 卡 scale-up 域）", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=1), "sync_dp", 1.0),
    ("美国·2 站同步 DP", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=2), "sync_dp", 1.0),
    ("美国·8 站同步 DP", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "sync_dp", 1.0),
    ("美国·8 站分层 AllReduce", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "hier_ar", 1.0),
    ("美国·8 站 DiLoCo H=30", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "diloco_h30", 1.0),
    ("美国·8 站 Streaming+4bit", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "diloco_h100_q4", 1.0),
    ("美国·24 站 Streaming+4bit", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=24), "diloco_h100_q4", 1.0),
    ("美国·8 站·EP 跨出 scale-up 域（反例）", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=512, n_sites=8), "diloco_h100_q4", 1.0),
    ("美国·8 站·PP 跨站（反例）", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "diloco_h100_q4", 1.0),
    ("美国·8 站·批量仅 1/4（关键批量受限）", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "diloco_h100_q4", 0.25),
    ("美国·8 站·批量 5.5×（Epoch 的 1.1B 假设）", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), "diloco_h100_q4", 5.5),
    ("中国·单站·小 scale-up 域（512）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=1), "sync_dp", 1.0),
    ("中国·单站·大 scale-up 域（4096）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=1), "sync_dp", 1.0),
    ("中国·8 站同步 DP（4096 域）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=8), "sync_dp", 1.0),
    ("中国·16 站 DiLoCo H=30（4096 域）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=16), "diloco_h30", 1.0),
    ("中国·32 站 Streaming+4bit（4096 域）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=32), "diloco_h100_q4", 1.0),
    ("中国·32 站 Streaming+4bit（512 域，对照）", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=32, n_sites=32), "diloco_h100_q4", 1.0),
    # —— 公平对照：把 scale-up 域当作架构变量而非国别属性 ——
    ("对照·美国加速器 + 4096 域·单站", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=1), "sync_dp", 1.0),
    ("对照·美国加速器 + 4096 域·8 站 Streaming", "central", "us_2030", 0.35,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=8), "diloco_h100_q4", 1.0),
    ("对照·中国加速器 + 1.6T scale-out·8 站 Streaming", "central", "cn_2030", 0.30,
     dict(tp=8, cp=2, pp=16, ep=256, n_sites=8), "diloco_h100_q4", 1.0),
]


def timeline():
    from params import SiteScenario
    rows = []
    for label, dk, ak, mfu_size, pc, stk, batch_mult in TIMELINE_CASES:
        d, a = DEMAND[dk], ACCEL[ak]
        strat = STRATEGIES[stk]
        site = SITES["us_2gw"] if label.startswith("美国") else SITES["cn_1gw"]
        n_dies = required_peak_flops(d, mfu_size) / a.die_fp8_flops
        # "小 scale-up 域" 对照：强制把域压到 512
        domain = a.scaleup_domain
        if "512 域" in label or "小 scale-up" in label:
            domain = 512
        if "4096 域" in label:
            domain = 4096
        if "1.6T scale-out" in label:
            net_scaleout_override = 1600e9
        else:
            net_scaleout_override = None
        cfg = ParallelConfig(dualpipe=True, micro_batch=1, **pc)
        net = NetworkHierarchy(
            scaleup_bw_bps=a.scaleup_bw_tbs * 1e12 * 8,
            scaleup_domain=domain,
            scaleout_bw_bps=net_scaleout_override or a.scaleout_bw_gbs * 1e9,
            wan_bw_bps=site.link_pbps * PB,
            wan_latency_s=site.prop_latency_s_per_hop,
        )
        r = simulate_step(
            n_layers=d.n_layers, d_model=d.d_model, seq_len=d.seq_len,
            n_params_total=d.n_params_total, n_params_active=d.n_params_active,
            global_batch_tokens=d.batch_tokens * batch_mult,
            topk=8, kv_frac=0.125, n_total_dies=n_dies,
            cfg=cfg, net=net, die_fp8_flops=a.die_fp8_flops,
            hbm_gb_per_die=a.hbm_gb_per_die,
            sync_every=strat.H, wan_bits_per_param=strat.bits_per_param,
            wan_fragments=strat.fragments,
            pp_crosses_wan=("PP 跨站" in label),
        )
        rows.append(dict(
            场景=label, 同步策略=strat.name,
            TP=cfg.tp, CP=cfg.cp, PP=cfg.pp, EP=cfg.ep, scaleup域=domain, 站点数=cfg.n_sites,
            总die数_万=round(n_dies / 1e4, 1),
            单副本die数=cfg.dies_per_replica, 副本数=round(r.n_replicas, 1),
            微批数=round(r.micro_batches, 1),
            单die每step_token=round(r.tokens_per_die_step, 1),
            计算_ms=round(r.t_compute * 1000, 2),
            TP_ms=round(r.t_tp * 1000, 2), CP_ms=round(r.t_cp * 1000, 2),
            EP_ms=round(r.t_ep * 1000, 2), PP_ms=round(r.t_pp * 1000, 2),
            站内DP_ms=round(r.t_dp_intra * 1000, 2),
            跨站DP摊薄_ms=round(r.t_dp_inter * 1000, 3),
            气泡_ms=round(r.t_bubble * 1000, 2),
            step_ms=round(r.t_step * 1000, 2), MFU=round(r.mfu, 3),
            单die参数_B=round(r.params_per_die / 1e9, 2),
            HBM压力_GB=round(r.hbm_pressure_gb, 1),
            HBM容量_GB=a.hbm_gb_per_die,
            HBM是否够=("是" if r.hbm_pressure_gb < a.hbm_gb_per_die * 0.6 else "否"),
            scaleup域是否放得下=("是" if r.scaleup_fits else "否"),
            跨站链路占用率=round(r.wan_link_util, 4),
            主要瓶颈=r.bound_by,
        ))
    write_csv("timeline.csv", rows)
    return rows



# ============================================================ 4b. 关键批量扫描
def batch_sweep():
    """关键批量大小（CBS）是 2030 推演中最大的单点不确定性。

    实测外推（arXiv:2410.21676 / 2412.01505 / 2505.23971）给出
    CBS ∝ D^0.26~0.47，外推到 10T token 只有数 M token 量级；
    而 Epoch 的 10GW 推演假设 1.1B token。两者差 2–3 个数量级。
    本扫描показ 这个差异如何直接决定 MFU 与可行站点数。
    """
    from params import SiteScenario
    rows = []
    d = DEMAND["central"]
    for path, ak, mfu_size, pc, domain, scaleout in [
        ("美国", "us_2030", 0.35, dict(tp=8, cp=2, pp=16, ep=32, n_sites=8), 576, 800e9),
        ("美国·大域", "us_2030", 0.35, dict(tp=8, cp=2, pp=16, ep=256, n_sites=8), 4096, 800e9),
        ("中国·大域", "cn_2030", 0.30, dict(tp=8, cp=2, pp=16, ep=256, n_sites=8), 4096, 400e9),
    ]:
        a = ACCEL[ak]
        site = SITES["us_2gw"] if path.startswith("美国") else SITES["cn_1gw"]
        n_dies = required_peak_flops(d, mfu_size) / a.die_fp8_flops
        strat = STRATEGIES["diloco_h100_q4"]
        for batch in [2e7, 5e7, 1e8, 2e8, 4e8, 8e8, 1.1e9, 2e9, 5e9]:
            cfg = ParallelConfig(dualpipe=True, micro_batch=1, **pc)
            net = NetworkHierarchy(
                scaleup_bw_bps=a.scaleup_bw_tbs * 1e12 * 8, scaleup_domain=domain,
                scaleout_bw_bps=scaleout, wan_bw_bps=site.link_pbps * PB,
                wan_latency_s=site.prop_latency_s_per_hop)
            r = simulate_step(
                n_layers=d.n_layers, d_model=d.d_model, seq_len=d.seq_len,
                n_params_total=d.n_params_total, n_params_active=d.n_params_active,
                global_batch_tokens=batch, topk=8, kv_frac=0.125, n_total_dies=n_dies,
                cfg=cfg, net=net, die_fp8_flops=a.die_fp8_flops,
                hbm_gb_per_die=a.hbm_gb_per_die, sync_every=strat.H,
                wan_bits_per_param=strat.bits_per_param, wan_fragments=strat.fragments)
            rows.append(dict(路径=path, 全局批量_Mtoken=round(batch / 1e6, 1),
                             单die每step_token=round(r.tokens_per_die_step, 1),
                             微批数=round(r.micro_batches, 1),
                             计算_ms=round(r.t_compute * 1000, 2),
                             站内DP_ms=round(r.t_dp_intra * 1000, 2),
                             step_ms=round(r.t_step * 1000, 2),
                             MFU=round(r.mfu, 3), 主要瓶颈=r.bound_by))
    write_csv("batch_sweep.csv", rows)
    return rows


# ============================================================ 5. 蒙特卡洛
def lognorm(rng, lo, hi, n):
    """给定 90% 区间 [lo, hi] 的对数正态采样。"""
    mu = (math.log(lo) + math.log(hi)) / 2
    sigma = (math.log(hi) - math.log(lo)) / (2 * 1.645)
    return np.exp(rng.normal(mu, sigma, n))


def montecarlo(n=40000, seed=20301018):
    rng = np.random.default_rng(seed)
    out = {}
    for path in ("美国", "中国"):
        # —— 不确定参数（90% 区间）——
        C = lognorm(rng, 3e28, 4e29, n)                    # 2030 单次训练算力 FLOP
        T = rng.uniform(70, 150, n) * 86400                 # 训练时长 s
        mfu = rng.uniform(0.22, 0.45, n)
        if path == "美国":
            eta = lognorm(rng, 5e12, 16e12, n)              # 设施级 FP8 FLOP/s per W
            psite = lognorm(rng, 0.8e9, 5e9, n)             # 单站功率上限 W
            link = lognorm(rng, 0.5e15, 10e15, n)           # 站点对带宽 bps
            dist = rng.uniform(300, 2000, n)                # 站间距 km
        else:
            eta = lognorm(rng, 1.5e12, 6e12, n)
            psite = lognorm(rng, 0.4e9, 2.5e9, n)
            link = lognorm(rng, 0.1e15, 3e15, n)
            dist = rng.uniform(800, 3000, n)
        nparams = lognorm(rng, 20e12, 200e12, n)            # 总参数量
        nactive = nparams / lognorm(rng, 10, 30, n)         # MoE 稀疏比
        ntok = lognorm(rng, 500e12, 4000e12, n)
        batch = lognorm(rng, 50e6, 1.2e9, n)                # 关键批量：这是最大的单点不确定性

        P = C / (T * mfu * eta)
        n_sites = np.ceil(P / psite)
        step_t = batch * 6 * nactive / (C / T)

        res = {"总功率_GW": P / GW, "站点数": n_sites, "step时间_s": step_t}
        for stk in ("sync_dp", "diloco_h30", "diloco_h100_q4"):
            st = STRATEGIES[stk]
            payload = nparams * st.bits_per_param / max(st.fragments, 1)
            budget = step_t * st.H * min(st.sync_budget_frac, 1.0)
            tprop = dist * 5e-6 * n_sites
            usable = np.maximum(budget - tprop, 1e-9)
            req = payload / usable
            res[f"{stk}_所需带宽_Pbps"] = req / PB
            res[f"{stk}_可行"] = (req <= link).astype(float)
        out[path] = res

    rows = []
    for path, res in out.items():
        for i in range(0, n, max(1, n // 4000)):   # 抽样写出，避免文件过大
            rows.append({"路径": path, **{k: float(v[i]) for k, v in res.items()}})
    write_csv("montecarlo.csv", rows)

    # 分位数摘要
    summ = []
    for path, res in out.items():
        for k, v in res.items():
            if k.endswith("可行"):
                summ.append(dict(路径=path, 指标=k, P10="", P50=f"{v.mean()*100:.1f}%", P90="", 均值=f"{v.mean()*100:.1f}%"))
            else:
                q = np.percentile(v, [10, 50, 90])
                summ.append(dict(路径=path, 指标=k, P10=f"{q[0]:.3g}", P50=f"{q[1]:.3g}",
                                 P90=f"{q[2]:.3g}", 均值=f"{v.mean():.3g}"))
    write_csv("montecarlo_summary.csv", summ)

    # —— 敏感性：对 log(站点数) 的 Spearman 相关 ——
    from scipy.stats import spearmanr
    sens = []
    for path in ("美国", "中国"):
        rng2 = np.random.default_rng(seed)
        C = lognorm(rng2, 3e28, 4e29, n); T = rng2.uniform(70, 150, n) * 86400
        mfu = rng2.uniform(0.22, 0.45, n)
        if path == "美国":
            eta = lognorm(rng2, 5e12, 16e12, n); psite = lognorm(rng2, 0.8e9, 5e9, n)
            link = lognorm(rng2, 0.5e15, 10e15, n); dist = rng2.uniform(300, 2000, n)
        else:
            eta = lognorm(rng2, 1.5e12, 6e12, n); psite = lognorm(rng2, 0.4e9, 2.5e9, n)
            link = lognorm(rng2, 0.1e15, 3e15, n); dist = rng2.uniform(800, 3000, n)
        nparams = lognorm(rng2, 20e12, 200e12, n)
        nactive = nparams / lognorm(rng2, 10, 30, n)
        ntok = lognorm(rng2, 500e12, 4000e12, n)
        batch = lognorm(rng2, 50e6, 1.2e9, n)
        P = C / (T * mfu * eta); ns = np.ceil(P / psite)
        step_t = batch * 6 * nactive / (C / T)
        st = STRATEGIES["diloco_h30"]
        payload = nparams * st.bits_per_param / max(st.fragments, 1)
        usable = np.maximum(step_t * st.H * st.sync_budget_frac - dist * 5e-6 * ns, 1e-9)
        req = payload / usable
        feas = (req <= link).astype(float)

        inputs = {"训练算力 C": C, "训练时长 T": T, "MFU": mfu, "设施能效 η": eta,
                  "单站功率上限": psite, "链路带宽": link, "站间距": dist,
                  "总参数量": nparams, "激活参数量": nactive, "训练token": ntok,
                  "关键批量 B": batch}
        for k, v in inputs.items():
            rho_n = spearmanr(v, ns).statistic
            rho_b = spearmanr(v, req).statistic
            rho_f = spearmanr(v, feas).statistic
            sens.append(dict(路径=path, 参数=k,
                             对站点数的相关=round(float(rho_n), 3),
                             对所需带宽的相关=round(float(rho_b), 3),
                             对可行性的相关=round(float(rho_f), 3)))
    sens.sort(key=lambda r: -abs(r["对站点数的相关"]))
    write_csv("sensitivity.csv", sens)
    return out, summ, sens


# ============================================================ 6. 故障率
def reliability():
    rows = []
    # Llama 3 标定：16384 GPU，419 次非预期中断 / 54 天 → 单卡 MTBI ≈ 5.8 年
    mtbf = 5.8
    for n_dies in [1.6e4, 1e5, 5e5, 1e6, 2e6, 4e6]:
        for label, elastic, quorum, ckpt, rec in [
            ("同步DP·传统检查点(30min)", False, 1.0, 1800, 600),
            ("同步DP·内存内检查点(1步)", False, 1.0, 10, 420),
            ("弹性DP(M=1)",             True, 0.95, 10, 420),
            ("Decoupled DiLoCo(M=8,K=1)", True, 0.875, 10, 60),
        ]:
            g = goodput(n_dies, mtbf, rec, ckpt, elastic, quorum)
            mtbi_min = (mtbf * 365.25 * 24 * 60) / n_dies
            rows.append(dict(加速器数=f"{n_dies:.0f}",
                             全局平均无故障间隔_min=round(mtbi_min, 2),
                             方案=label, goodput=round(g, 3)))
    write_csv("reliability.csv", rows)
    return rows


# ============================================================ main
def main():
    print("[1/6] 模型校验（对照 Epoch AI 10GW 推演）")
    val = validation()
    print("[2/6] 主场景扫描")
    sc = scenarios()
    print("[3/6] 带宽扫描")
    bw = bandwidth_sweep()
    print("[4/6] 通信时间线仿真 + 关键批量扫描")
    tl = timeline()
    bs = batch_sweep()
    print("[5/6] 蒙特卡洛 + 敏感性")
    mc, summ, sens = montecarlo()
    print("[6/6] 故障率与 goodput")
    rel = reliability()

    # 汇总 JSON
    import numpy as _np
    summary = {
        "validation": val,
        "scenarios": sc,
        "bandwidth_sweep": bw,
        "timeline": tl,
        "batch_sweep": bs,
        "montecarlo_summary": summ,
        "sensitivity": sens,
        "reliability": rel,
    }
    with open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print("  wrote data/summary.json")


if __name__ == "__main__":
    main()
