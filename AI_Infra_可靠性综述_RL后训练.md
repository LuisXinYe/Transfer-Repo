# AI Infra 可靠性:业界与学术界工作梳理
## —— 以 RL 后训练、尤其是 Agentic RL 后训练为中心

> 整理日期:2026-09-22
> 范围:大模型训练/后训练/推理的可靠性(容错、故障检测与诊断、checkpoint、弹性、goodput 工程)，重点落在 **RL 后训练** 与 **Agentic RL 后训练**。
>
> 出处说明:§8 参考文献分为两部分。「2026 新工作」均已联网核对 arXiv/USENIX/仓库原文;「2025 及更早」来自既有整理，会议名与年份可能有偏差，引用前请核对。

---

## 0. 一页速览:这份材料怎么用

| 你要解决的问题 | 先读这几篇 |
|---|---|
| RL 后训练整体容错架构怎么切 | RobustRL(角色级容错)、Belayer(agentic 专属)、slime 容错文档 |
| Agentic rollout 中断了怎么不丢轨迹 | Belayer(engine 侧 KV 重建 + env 侧 checkpoint-restore)、RobustRL §5.2.2、AReaL partial rollout |
| 沙箱/环境这一层的可靠性 | The Rollout Infrastructure Tax、LiteResearcher、Orchard、Belayer |
| 慢节点 / fail-slow(RL 里最致命) | ARGUS、SCOUT、Guard、Minder、Greyhound |
| 权重同步失败与一致性 | RobustRL §5.2(UCX + relay)、SparseRL-Sync、AReaL-DTE、ROSE |
| 资源弹性/借机器 | Weave、ROSE、Libra、BiDiRL、RobustRL §5.1.3 |
| 不是硬件故障、是"训练跑飞了" | RFT-FaultBench / RFT-FM |
| 国产卡(Ascend/NPU)场景 | Libra(160×910B3)、MindSpeed RL、SIGMA(early-life hardware)、MindIO/MindCluster |

**一句话判断:2024-2025 的可靠性工作主战场是预训练(SPMD、同构、单一失败模式);2026 的主战场明显迁移到了 RL / Agentic RL 后训练——因为这里同时存在训练故障、推理故障、环境与工具故障**三类来源，而前两代系统只处理前两类。

---

## 1. 为什么 Agentic RL 后训练的可靠性是一个新问题

先把差异说清楚，后面所有方案的取舍都源于这张表。

| 维度 | 预训练 | RL 后训练 | Agentic RL 后训练 |
|---|---|---|---|
| 执行模型 | SPMD，全同构 | 多角色(trainer / rollout / 管理面) | 多角色 + 有状态、长时、交互式 rollout |
| 故障影响半径 | 全 job | 单角色可隔离 | 单角色 + 单环境实例 + 单轨迹 |
| "GPU 无活动" 语义 | 大概率是挂了 | 可能是在等 rollout | 大概率是在等工具/环境返回，误判率极高 |
| 关键路径通信 | NCCL/HCCL | NCCL + 权重同步(UCX/RDMA) | 再加 HTTP 工具调用、沙箱 RPC、Store 读写 |
| 状态载体 | 模型权重 + 优化器 | + 采样进度 | + 多轮轨迹、KV cache、容器文件系统状态 |
| 重启代价 | 回到上一个 ckpt | 回到上一个 ckpt + 重跑 rollout | 上面全部 + 重放数十次工具交互(最贵) |
| 新增故障源 | 硬件、网络 | 推理引擎 hang、权重同步 | 沙箱冷启动/泄漏/超时、外部 API 限流、奖励评测服务 |

> Agentic RL 的本质困难:rollout 从无状态的一次前向，变成了有状态的长事务。传统"重启 worker"这个动作在这里等于"回滚一个跑了 10 分钟、调了 30 次工具的事务"。所有 2026 的 agentic 容错工作，核心都在回答"如何不回滚这个事务"。

---

## 2. 核心板块:RL / Agentic RL 后训练的可靠性

### 2.1 恢复动作的层次与关系

`TokenRetry → InstanceRestart → StepRetry → ProcessRestart → JobRestart`，读起来像九级线性阶梯。但把 6.2–6.6 各节末尾的「升级」原文拼起来是这样:

| 节 | 升级到 |
|---|---|
| 6.2 TokenRetry | 多次重推失败 → InstanceRestart |
| 6.3 InstanceRestart | 多次无法恢复 / 健康节点不足 → 直接上报 ClusterX 进编排层 |
| 6.4 StepRetry | 失败 / MindIO 无法原地修 → ProcessRestart / JobRestart |
| 6.5 ProcessRestart | 失败 → JobRestart |
| 6.6 JobRestart | 仍失败 → 上报 ClusterX |

`InstanceRestart` 失败后不会进 `StepRetry`，它直接上浮编排层。 真实结构是**两条平行链**:

```
推理侧   TokenRetry ──▶ InstanceRestart ─────────────┐
                                                     │
训练侧   StepRetry ──▶ ProcessRestart ──▶ JobRestart ─┤
                                                     ▼
                                    ┌─ 编排层(ClusterX 决策)─┐
                                    │ StoreRestart           │
                                    │ JobResubmit            │
                                    │ PodReschedule          │
                                    │ ClusterRecreate        │
                                    └───────────┬────────────┘
                                                ▼
                                            人工介入
                          (AgenticRLJob CRD 始终不重建 = 不动点)
```

分成两条是**对的(推理与训练是两个子系统，一个挂了不该去重试另一个)，只是总结图画错了**，建议改为两条线。

最清晰的心智模型:每一档 = 一条「重建什么 / 保留什么」的切线，越往下切得越深:

| 动作 | 重建的对象 | 保留下来的 |
|---|---|---|
| TokenRetry | 一次 token 推送 | 一切 |
| InstanceRestart | 一个 vLLM 进程(冷启动) | 训练侧全部 + 其他实例 |
| StepRetry | 一个训练 step 的计算 | 进程不杀、通信域、权重 |
| ProcessRestart | FSDP Worker 进程 + HCCL 通信域 | Driver、内存中的 `training_state`、全部 vLLM |
| JobRestart | Driver + 全部训练进程 | Pod、RayCluster、Store |
| StoreRestart | Store Pod ＋ 连带删/重建 RayJob | Pod 调度、RayCluster |
| JobResubmit | RayJob(提交链路) | RayCluster、WorkerPod |
| PodReschedule | 一个 WorkerPod(换节点) | RayCluster 本体 |
| ClusterRecreate | 整个 RayCluster | AgenticRLJob CRD |

两条分界线:

- ProcessRestart / JobRestart 之间 = **Driver 在不在。Driver 还在 → 内存里的 `training_state` 还在;没了 → 必须从 Store/NFS 读回。这就是"状态外置"整件事的存在理由。**
- JobRestart / 编排层之间 = **Pod 动不动**。进程内恢复不碰 K8s 资源;编排层要删 Pod、等调度、拉镜像、重新图编译，量级完全不同。

**两个排序陷阱**:

1. StoreRestart 的名字骗人——流程第 3 步「删除 RayJob，保护训练数据完整性」，它实际包含一次 JobResubmit，论影响半径 `StoreRestart ≳ JobResubmit`，排序是反的。建议改为**有界本地缓冲 + 背压:写入方缓冲一个有界窗口(如 60s / N MB)，满则阻塞上游;Store 恢复后排空。只有停机超过该窗口才升级到删 RayJob——既保证数据完整性(数据被持有而非丢弃)，又切断"秒级故障 → 分钟级重建"的放大链。且数据库型 Store 自带持久化，更不该照删**。
2. **编排层四档是并列不是递进——StoreRestart / JobResubmit / PodReschedule 对应四种不同触发场景**，只有 ClusterRecreate 兼具兜底含义;而进程内两条链才是真正的递进(同一对象逐级加码)。建议文档用不同符号区分(递进 `→`、并列 `|`)，现在统一用箭头必然被误读成九级阶梯。

---

### 2.2 角色级容错 —— RobustRL 

Role-Based Fault Tolerance System for LLM RL Post-Training，arXiv:2512.22492(2025-12-27)，浙大等。

- **核心范式:Detect → Restart → Reconnect。把 trainer / rollout / 管理角色当作独立的分布式子任务**，只重启故障角色并重连存活角色，消除全量重启(含 rollout 重放与初始化)的开销。
- **角色与阶段感知的检测**:trainer 侧仅在训练阶段判定"TensorCore 活动为 0 持续 5 分钟";rollout 侧判定"TPS 为 0 持续 60 秒 + 心跳确认"。
- **Robust trainer**:step 函数 try-catch;所有 trainer 一起重启(通信域整体重建);ByteCheckpoint 逐 step 落盘(GPU→内存 ~3s 阻塞，内存→磁盘 ~10s 异步);全任务重启的兜底规则(首迭代失败 / 同一 step 重复失败 / 重启反复失败)。
- 借 rollout 机器给 trainer 热身(§5.1.3):kill 一个 rollout，把机器秒级交给 trainer。前提是同机房同构机器，并预留 max(DP，TP，PP，EP，CP) 个 rollout。
- **权重同步(§5.2.1):reshard → trainer DP 组按 rank 点对点推给 rollout 副本，GPU 上 torch tensor 经 DLPack 零拷贝转 cupy，走 UCX;relay 中继**——已更新的 rollout 转为中继服务器，扩展性接近对数。
- rollout 故障处理(§5.2.2):RequestManager 按工具轮次保留轨迹;中继服务器故障则换中继并从进度续传;拉权重时 trainer 故障则清理半份权重并等待。

**可借鉴点**:角色级隔离的思想、阶段感知的检测阈值、升级到全任务重启的三条判据（首迭代异常、重复异常、重启反复失败）、relay 式权重分发。

---

### 2.3 Agentic 专属容错 —— Belayer

Belayer: Efficient Fault Tolerance for LLM Agentic RL Training，arXiv:2608.14635(2026-07-28，v2 2026-08-18)，Jiecheng Zhou, Qinghao Hu, Peng Sun, Xingcheng Zhang, Weiming Zhang。

这是目前唯一正面处理"agentic RL 的有状态 rollout 故障"的系统工作。它的问题陈述就是上表那句话:*agentic RL 有状态、交互式的执行模型，把熟悉的故障变成了新的恢复问题——rollout worker 故障会中断有状态的 in-flight 请求*。

两条恢复路径:

**路径一:rollout engine 故障 —— shadow worker 接管**

机制建立在一个前提上:**把权重与 KV arena 的所有权从 worker 进程剥离出去**。二者由独立的 weight server / KV-cache server 申请并持有，worker 只通过 CUDA IPC handle 映射使用。数据始终在 HBM，位置不变，变的是生命周期归属——worker 进程崩溃时，驱动回收的只是它的映射，实际分配仍由 owner 持有，不被释放。

在此之上，每个 rollout engine 配一个预热好的 shadow 进程，shadow 与主 worker 位于同一张卡，shadow 进程建 CUDA context、准备 CUDA graph buffer、各种运行时初始化，它不持有权重也不持有 arena(二者均为映射)，自身仅占 **1.84GB** 显存，平时不承担任何计算，只监测主 worker 的存活。

- **接管(五步)**:① shadow 经心跳发现故障;② 执行 GPU 健康检查、确认 weight/KV server 存活、flush KV pool;③ 重映射 CUDA-graph buffer 并就绪;④ 通知 router 注销故障 worker、注册自己;⑤ router 改路由，shadow 经 token 级上下文恢复重建 KV 并续跑。耗时约 **1 秒**。

**收益的来源**:冷启动必须执行四项高成本步骤——加载权重、profile 探测激活峰值、分配 KV arena、**捕获 CUDA graph**。前三项 shadow 完全跳过(权重与 arena 由 owner 持有，shadow 在待命期间已持有独立映射，接管时零动作);第四项由**重新捕获**降级为**重映射已有 buffer**——CUDA graph 会把设备指针烘焙进图中，而 shadow 是另一进程、地址空间不同，故需经虚拟内存管理器做一次地址对齐，代价是微秒到毫秒级，而重新捕获是秒级到分钟级(昇腾上尤甚)。此外仅需重建 worker 本地态:进程上下文、CUDA context、通信组、请求调度器状态。据此恢复速度较冷启动快 **42×**，单次故障对训练时间的影响为 **+1.16%**，而冷启动基线为 **+8.72%**。

**可借鉴点**:

对照你们的 `InstanceRestart`(§6.3)，差异可以逐步对上:

| | 你们现有流程 | shadow 方案 |
|---|---|---|
| 1 | 检测心跳超时，上报 SDK 决策 | 同(改为 mtime 心跳，由 shadow 本地监测) |
| 2 | LoadBalancer 移除故障实例 | 同(router 注销) |
| 3 | **健康节点重建 vLLM Server** | **不存在** |
| 4 | **CheckpointEngine 推送最新权重** | **不存在** |
| 5 | 新实例加回负载均衡 | 同(router 注册 shadow) |

第 3、4 步整个消失，这就是分钟级变 1 秒的直接原因——shadow 在待命期间已持有对权重与 arena 的独立映射，接管时无需加载、也无需映射。

此外有一项 RL 特有的正确性收益:冷启动重建的实例，第 4 步推送的是**当前最新版**权重。若被中断的轨迹前半段由 $W_k$ 生成，而重建后实例拿到 $W_{k+1}$，则同一条轨迹内权重版本不一致，重要性采样比失真——slime 要求"参数更新到正确版本后再接请求"正是为此。shadow 接管使用的是同一块内存中的同一份权重，**版本天然一致，无需校验**。

其余两点:

- "环境 checkpoint 必须和模型 context 一起回滚"这条一致性约束，与上述轨迹内权重版本一致性是同一类不变量。
- 用推理空闲期做 checkpoint 的 overlap 策略，可直接用于你们 per-step 异步 checkpoint 的调度。

**路径二:环境/沙箱故障 —— full checkpoint-restore**
对容器文件系统 + 运行时状态做完整 checkpoint-restore，并与 LLM 侧 context 协同以保持 prefix 一致性(这一步是关键:环境回滚到 t，模型 context 也必须回到 t，否则轨迹自相矛盾)。
自适应策略:当预测的间隔足以摊销开销时，把 full-state checkpoint overlap 到 LLM 推理的空闲期。
效果:环境故障恢复 **快 1.5–3.5×**，无故障期开销低。

---

### 2.4 框架级工程实践:心跳 + 路由摘除(slime / GLM-5)

**GLM-5 技术报告(arXiv:2602.15763)与 slime** 框架(THUDM，GLM-4.5 至 GLM-5.3 的 RL 底座)。

slime 的容错文档(`docs/zh/advanced/fault-tolerance.md`)给出了一套非常"能落地"的最小可用方案:

- **覆盖范围**:只管 rollout engine 故障(SGLang server hang、长尾样本拖慢轮次)。显式声明:集群抢占、trainer rank 故障、全 job resume 交给调度器 + Ray restart policy + checkpointing。
- **检测**:对所有 SGLang server 周期性发 `/health_generate` 心跳。
- **恢复:心跳超时 → 停掉异常 server → 当前 rollout 轮次结束后重启 → 先更新到正确的权重版本**再接新请求。
- **配置**:`--use-fault-tolerance`;`--rollout-health-check-first-wait`(默认 300s，给 kernel 编译留时间)、`--rollout-health-check-interval`(10s)、`--rollout-health-check-timeout`(5s)。
- 还有 `--debug-rollout-only` / `--save-debug-rollout-data` / `--load-debug-rollout-data` / `--debug-train-only` 这组**故障复现开关**。

GLM-5 报告侧的表述:rollout server 周期发心跳由编排层监控，不健康的 server 主动终止并从 inference router 注销，重试自动路由到健康 server，避免单 server 事故打断整轮 rollout。

**可借鉴点(强烈建议)**:
- `first-wait` 这个细节——首次心跳要等 300s，因为引擎启动含 kernel 编译。你们 vLLM on Ascend + CANN 图编译的首启更慢，这个坑一定会踩。
- "从 router 注销"比"重启实例"更重要:你们的 LLMProxy + LoadBalancer 目前在故障表里是缺失的，而它恰恰是实现"故障 rollout 实例零影响"的那个开关。
- 那组 debug 开关值得直接抄:能把 rollout 和 train 拆开单独复现，是做故障注入 CI 的前提。

---

### 2.5 训练动力学故障(不是硬件挂了，是"跑飞了")—— RFT-FaultBench / RFT-FM 

Towards Robust LLM Post-Training: Automatic Failure Management for Reinforcement Fine-Tuning，arXiv:2605.04431(2026-05-06)，Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Liancheng Fang, Kening Zheng, Hongyi Liu, Xiaosong Huang, Philip S. Yu, Ying Li。

**RFT-FaultBench 是首个 RFT 细粒度故障基准，5 族 / 16 型**:

| 族 | 类型 | 含义 |
|---|---|---|
| RF 奖励 | RF-1 Reward Spike | 奖励膨胀到超出质量对齐的正常区间 |
| | RF-2 Reward Collapse | 奖励信号被压制或趋近于零 |
| | RF-3 Reward Hacking | 策略钻奖励漏洞而非真正解题 |
| PG 策略生成 | PG-1 Empty Response | 输出空白/平凡/近乎无内容 |
| | PG-2 Repetition Collapse | 退化成重复 token、短语或循环 |
| | PG-3S / PG-3L Length | 响应异常短 / 异常长、拖尾、长度失控 |
| OD 优化动力学 | OD-1 KL Explosion | 更新过激，偏离 reference |
| | OD-2 Update Freeze | 训练仍在跑，但有效更新几乎消失 |
| | OD-3 Entropy Collapse | 熵下降过快，探索坍塌 |
| CA 信用分配 | CA-1 Value Mismatch | value 估计与实际回报长期不一致 |
| | CA-2 Advantage Instability | 优势估计噪声过大/不稳 |
| | CA-3 Delayed Credit | 延迟奖励在动作/步之间错误归因 |
| TE 工具/环境 | TE-1 Tool Call Error | 调用格式错误或执行失败 |
| | TE-2 Observation Corruption | 观测在到达策略前被污染 |
| | TE-3 Termination Error | episode 错误或过早结束，轨迹被截断 |

**RFT-FM** 把 异常检测 → 故障诊断 → 自动修复 串成闭环:
- 检测:不用绝对阈值，而是从健康跑校准 **normal profile**，做**偏差式不变量提取**。五个 RFT 专属不变量:奖励轨迹一致性、KL 动力学、熵剖面、回报稳定性、生成质量。遥测源为 reward / KL / entropy / returns / response length / policy loss / 工具交互信号。
- 诊断:时序动力学建模 → 细粒度**故障指纹** → 映射到族级/型级标签。
- 修复:用 **qwen-plus** 做 diagnosis-grounded 推理与动作规划，改训练配置后回验严重度。

**实验规模必须看清**:8×H20、OpenRLHF、一个"受控的算术式推理任务";779 次运行中 768 次为注入、11 次正常基线。这是玩具任务上人工注入的基准，不是生产 trace，故障指纹能否迁移到千卡长轨迹 agentic RL 无证据。

结果(作者很诚实地不好看):

| | Easy | Hard |
|---|---|---|
| 检测 F1 | 87.96%(P 99.60% / R 78.75%) | 73.88%(P 99.62% / R 58.71%) |
| 诊断 Macro-F1 | 91.46% | 50.16% |

- precision ~99.6% / recall 58.7% 是刻意的设计点:宁可漏报不误报(生产系统里的正确取舍)，但代价是困难故障漏掉四成。
- 诊断 hard 设定下 50%，接近抛硬币。**CA 族最差**(检测 easy F1 仅 53.17%，诊断 hard 39.34%)，信用分配故障可观测性最低。
- **自动修复建议不采信**:整体缓解率 **46.25%**，严重度变化中位数 **−5.84%**;作者承认 one-shot 干预经常失败、"从诊断到干预闭环可行但远未解决"。
- 基线对比:检测优于 Isolation Forest / LOF / TranAD / OmniAnomaly / Anomaly Transformer;诊断优于 KNN / SVM / RUN / CausalRCA / CIRCA。

一个对 agentic 场景有利的发现:按族看修复效果，TE(工具/环境)最好，缓解率 73.33%，检测 F1 90.53%;OD(优化动力学)最差，26.67%。原因是环境/工具故障机械、可观测、处置确定(重试/换沙箱/丢弃轨迹)，而熵坍塌类要动超参且反馈滞后。即 agentic 场景最高频的一族恰好最好治，应优先做。

**为什么重要:这是唯一一篇把"RL 后训练的可靠性"从系统故障**扩展到**算法故障**的工作。**可靠性指标不能只有 ETTR**——一个 ETTR 99% 但 reward 已跑飞 300 step 的任务，浪费的算力比宕机严重得多。

**可直接拿走的三样**:(1) **TE-1/2/3 三分法补进故障分类与映射表;(2) 五个不变量当监控清单(一周可实现，不需要它那套模型);(3) "相对 normal profile 的偏差"而非绝对阈值**——与 Minder(LSTM-VAE 找 outlier 机器)、Greyhound(变点检测)是同一方法论，只是对象从硬件指标换成训练动力学，两边可共用基线建模与告警通道。
**不该拿的**:它的自动修复闭环，以及把它的实验数字当作自己系统能达到的依据。

---

### 2.6 解耦 / 异步架构:可靠性的红利与新风险

这一批 2026 工作主要目标是**吞吐**，但架构选择直接决定了故障半径，所以必须一起看。

| 工作 | 出处 | 架构要点 | 对可靠性的影响 |
|---|---|---|---|
| RollArt | arXiv:2512.22560(2025-12，2026-06 修订)，阿里 | 解耦式多任务 agentic RL:prefill 用算力型 GPU、decode 用带宽型 GPU、环境执行放 CPU 集群;轨迹级解耦，生成/环境交互/奖励打分各自独立推进 | 直接结论:"慢的或失败的环境永远不阻塞其他环境"。故障隔离从"实例级"下沉到"轨迹级"。3000+ GPU 阿里集群、Qoder 产品的数千亿 MoE 上验证，1.31–2.05× 训练时间下降 |
| Weave | OSDI '26，HKUST + UIUC + 阿里(Tianyuan Wu 等) | rollout/training 物理解耦后，on-policy 同步产生严重 dependency bubble;用跨集群编排把 A 任务的结构性空闲喂给 B 任务的活跃阶段 | 两周生产 trace 回放:同等负载下供给成本比 veRL 低 1.38×、比朴素解耦低 1.84×，且 100% SLO 达成。把"空闲"当资源看，和"把故障机器当资源看"是同一套调度机制 |
| ROSE | arXiv:2605.06534(2026-05)，阿里/HKUST | 协同弹性:把在线 serving 集群已部署但空闲的 GPU 借给 rollout。SLO-safe 协同执行器 + 跨集群权重传输引擎(shard-aware 路由 + 权重稀疏)+ 弹性 rollout 调度器 | 1.3–3.3× 吞吐，rollout 时间降 1.2–1.5×，零 SLO 违约。关键副作用:serving GPU 会随时加入/离开，这逼迫权重传输必须是容错的点对点，不能用固定集合通信域 |
| Libra | arXiv:2606.03077(2026-06，v3 2026-09)，CUHK 等 | 因果引导的分桶调度器(把请求路由到不同并行配置的执行桶，治长尾 straggler)+ 跨阶段非阻塞地在 rollout/training 之间动态重分配 worker | 48×A800 与 160×Ascend 910B3 双平台验证，4.2× 吞吐、2.7× reward 收敛加速。"非阻塞地搬 worker 且不破坏进行中的训练"正是 RobustRL §5.1.3 借机器那套的推广 |
| BiDiRL | arXiv:2607.09207(2026-07) | hot-switch 运行时 + 调度感知规划器 + 双向运行时调度器，在 rollout 与 training 之间双向回收空闲 | 32-GPU 上 1.94×(vs veRL/AReaL/ROLL)，不影响收敛。hot-switch 的低开销切换能力可复用为"故障后快速再配置" |
| OrchestrRL | arXiv:2601.01209(2026-01) | 自适应并行度调度 + RFabric:用光电路交换机按需重构汇聚层/核心层，为 training/generation/权重同步三种流量分别定制带宽 | 64×H800 上 1.42×。网络可重构意味着链路降级时可以绕行，这是一条新的容错维度 |
| Laminar | EuroSys 2026，arXiv:2510.12633，ByteDance Seed + HKU(verl/HybridFlow 一作班底) | 针对轨迹生成的极端长尾偏斜:①轨迹级异步，每条轨迹按自己的节奏独立生成与消费;②完全解耦架构;③relay worker 层作为分布式参数服务，取代全局权重同步，rollout 可随时拉最新权重而不阻塞 actor 训练循环;④dynamic repack 把长尾轨迹归并到少数专用 rollout | 1024 GPU 上吞吐最高 5.48×，收敛时间缩短。可靠性是解耦的副产品(故障半径降到单条轨迹)，非专门机制。两个陷阱:(a) 此处的 relay 与 RobustRL 的 relay 不是一回事——RobustRL 是"已更新 rollout 转中继"解带宽扩展性，Laminar 是"专用参数服务层"解时间解耦，二者正交可叠加;(b) dynamic repack 会吃掉 fail-slow 信号，因为"长轨迹"与"慢实例"外观相同，埋点必须放在 repack 之前，且用按 token 归一化的 TPS 而非轨迹墙钟时间 |
| StreamRL / AReaL | 2025 / NeurIPS'25 | 全异步、生成与训练解耦;AReaL 的 partial rollout 缓解同步屏障 | 异步天然容错更好(单 rollout 挂了不阻塞训练)，但引入 staleness 与轨迹内权重版本不一致;AReaL 的 partial rollout 有轨迹中止/续跑 + KV 重算的开销。相关补窟窿工作:Staleness-Constrained Rollout Coordination(arXiv:2601.12784)、Periodic Asynchrony(arXiv:2511.18871) |

> **对你们方案的判断:你们当前 colocated 模式是这几档里容错能力最弱**的——一个进程挂掉同时打掉训练和推理。如果中期要往 agentic 长轨迹走，解耦(至少是 rollout 独立 Pod)几乎是必选项，RollArt 的"轨迹级解耦"是目标形态。

---

### 2.7 被严重低估的一层:环境与沙箱 

Agentic RL 里 rollout 时间的大头常常不在 GPU 上，而在沙箱里。

- The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning，arXiv:2607.01415(2026-07-01)，Daniel Thi Graviet, Lovre Pesut, Ivan Dagelic, Vedran Jukic, Ivan Burazin。
  对四种执行基座做对比:单容器 / 托管沙箱 / K8s 编排容器 / 云 VM。结论:**冷启动延迟差异高达 110×;换算到"100 万条 150 步轨迹"的总 worker-hours，差异 1.8×**。作者主张:*未来的 coding-agent RL 系统应当把执行基座当作训练系统的一部分来优化，而不是部署管道的附属品*。
- **Orchard(arXiv:2605.15040):沙箱每条命令执行延迟 0.28s 是关键指标，因为每条 rollout 要几十次环境交互，训练吞吐直接随沙箱响应性线性缩放**。
- **LiteResearcher(arXiv:2604.17931):在线环境不稳定是 reward noise 的主要来源**;改用确定性本地环境消除该噪声后，可稳定跑 700+ RL step 且单调提升。
- **EnvFactory(arXiv:2605.18703)、CodeMidas**(arXiv:2609.22068):从代码/可执行环境合成角度扩展环境规模，侧面说明环境供给本身已成瓶颈。

**可借鉴点**:
- 你们的故障分类表(`4.1` 九个故障对象)全是基础设施，无「环境/工具」行——环境故障只作为推理实例故障的副作用被兜底链顺带接住，未作为独立故障对象。建议单列一族:沙箱冷启动超时、沙箱 OOM/磁盘满、外部 API 限流/5xx、工具返回格式损坏、奖励评测服务不可用。详见 §7.1。
- 环境故障的正确处置通常不是重启 Pod，而是丢弃该条轨迹并补采——前提是你的 batch 组装逻辑能容忍"少了几条"，这是一个需要在 Store/Daemon 层显式设计的能力(可配的 `min_valid_trajectories` 阈值 + 补采)。
- LiteResearcher 的发现提醒:环境不稳定会被算法误读成 reward 噪声，所以环境失败率必须作为一等指标上报，而不是吞掉重试。

---

### 2.8 权重同步的可靠性 

权重同步是 RL 后训练独有的、横跨两个通信域的关键路径，也是单点最多的地方。

| 方案 | 要点 |
|---|---|
| RobustRL §5.2.1 | UCX 点对点 + DLPack 零拷贝 + relay 中继(已更新的 rollout 转为中继)，235B/470GB 实测 ~6s;中继故障可换中继续传 |
| SparseRL-Sync | arXiv:2605.07330。用稀疏 (I，V) 消息只传变化的索引与新值，替代全量权重广播，通信量 ~100× 下降，已集成 slime |
| AReaL-DTE | arXiv:2608.00455。免快照的稀疏策略权重传输，面向训练与 rollout 推理微服务之间的在线 agentic RL |
| ROSE 权重引擎 | 跨集群、shard-aware 路由 + 权重稀疏;因为 serving GPU 动态进出，必须是容错的点对点传输，不能用固定集合通信组 |

**可借鉴点:通信量降 100× 本身就是最好的可靠性手段(传输窗口从 6s 降到 0.1s 量级，被故障打断的概率同比下降)。另外 ROSE 那条结论对你们直接适用:只要 rollout 实例数是弹性的，权重同步就不能建在固定 HCCL 通信域上**。

---

## 3. 底座:通用大模型训练可靠性(RL 场景仍然复用)

### 3.1 故障表征与度量

| 工作 | 出处 | 关键数据 |
|---|---|---|
| Llama 3 训练报告 | Meta, 2024 | 54 天内 466 次中断，78% 归因硬件 |
| Revisiting Reliability in Large-Scale ML Research Clusters | Meta, HPCA'25 (arXiv:2410.21680) | 提出并系统化 ETTR(有效训练时间比) |
| Acme | NSDI'24，上海 AI Lab | 生产集群故障特征与分类 |
| Philly | ATC'19，微软 | 早期 DL 集群故障分析的奠基工作 |
| What-if analysis | OSDI'25 | 对算子时间线做反事实回放 |
| From Detection to Recovery(504 GPU 运维报告) | arXiv:2605.09370(2026-05)，SKT / Upstage / Lablup / NVIDIA Korea / VAST Data | 63 节点 B200、55 天指标 + 73 天日志、224 次训练会话。核心结论:分析 751 个 Prometheus 指标与 10 次 GPU 故障后，没有任何单一指标能跨故障类型稳定主导，必须多信号检测;重启加载只用到峰值读带宽的 21.5%(700GB/s)，保存只用到写带宽的 16.0%(250GB/s);63 个节点里 top-3 占了 50%+ 的排除次数(故障高度集中，支持节点黑名单策略);自动重试成功率 33.3%(12 条重试链 / 73 次尝试)，是人工重试 12.5% 的 2.7×;重试间隔中位数 11 分钟 |
| SIGMA | arXiv:2512.13488(2025-12)，MSRA | 面向 early-life 加速器(新硬件、失败模式未定义)的训练栈:Lucia 训练平台 LTP + 训练框架 LTF。2048 卡 / 75 天训 200B MoE，有效利用率 94.45%、MFU 21.08%，全程仅 1 次稳定性事件。国产卡场景最可对标的一篇 |
| SDC 现状 /| Meta 2021 / Google HotOS'21 / OCP SDC-in-AI 白皮书 | 大致口径:约 1/1000 机器受静默数据损坏影响;大规模 AI 训练大约 每 1–2 周遭遇一次 SDC 事件 |

### 3.2 检测与诊断(2026 新增了三篇重要工作)

| 工作 | 出处 | 机制 |
|---|---|---|
| ARGUS | arXiv:2606.20374(2026-06) | 常开、细粒度的生产级 tracing。三层分解:CPU 调用栈 / 框架语义 / GPU kernel 执行，合计开销 <2%(对比细粒度 profiler 的 5–30%);kernel 事件压缩 ~3700×(10MB → 2.7KB per rank per step)。渐进式诊断:iteration 级 → phase 级 → kernel 级，逐层隔离异常窗口、straggler rank、退化 kernel。在 10,000+ GPU 生产集群连续部署 6 个月以上。覆盖算力 straggler、链路降级、流水气泡放大、FlashAttention JIT 卡顿、被通信症状掩盖的算力 straggler。统计:59% 的 512–1024 GPU 作业遭遇 fail-slow，平均 JCT 延长 34.59% |
| SCOUT | arXiv:2608.11034(2026-08)，Zhuang Wang | 单一设计原则:在等价副本之间用严格多数共识识别 outlier。核心机制 C3(Consensus Collective Communication)——找出紧凑签名偏离同伴共识的 rank。在线定位 hang / straggler / SDC 三类。无需改代码即可接入 PyTorch、TorchTitan、Megatron-Core、DeepSpeed。开源:github.com/LMResiliency/lm-resiliency |
| Guard | arXiv:2605.17879，MLSys 2026 | 双相检测:训练中的轻量在线性能监控 + 上线前的离线 node-sweep 节点资格评估。效果:FLOPs 利用率最高 1.7×，run-to-run step 方差从 20% 降到 1%，MTTF 提升 |
| L4 | arXiv:2503.20263 | 自动化日志分析诊断大规模训练故障 |
| Minder | NSDI'25 | LSTM-VAE 做指标嵌入，用两两距离找 outlier 机器 |
| Mycroft | 2025 | NCCL proxy/transport 插桩 + 依赖链根因定位 |
| MegaScale | NSDI'24，字节 | 全栈监控 + 诊断，万卡训练 |
| ByteRobust | 2025，字节 | 预训练鲁棒性系统:job 级重启、10s GPU / 30s 网络巡检、py-spy 栈聚类、并行组驱逐、热备池、热更新。20 万+ GPU 规模;9600 GPU 三个月任务 ETTR 97% |
| C4 | SIGCOMM'24，阿里 | 通信驱动的故障定位 |
| Greyhound / FALCON | ATC'25 | BOCD 变点检测做 fail-slow |
| SuperBench | ATC'24，微软 | 上线前主动验证，proactive validation |
| TRANSOM | 2023 | 端到端容错训练系统 |
| TrainCheck | OSDI'25 | 用训练不变量捕获静默错误 |
| TrainVerify | SOSP'25 | 执行计划等价性验证 |
| 工具层 | — | DCGM(nv-hostengine、NVML + profiling counter、`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`、`dcgmi health`、`dcgmi diag -r 1~4`、dcgm-exporter);PyTorch NCCL Flight Recorder(每 rank 集合通信环形缓冲，watchdog 超时时 dump，跨 rank 比对 seq);NCCL RAS;py-spy |

> **RL 场景的关键缺口:上述诊断工具几乎全部建立在 "同构副本 + NCCL 为唯一关键路径" 的假设上。SCOUT 的"等价副本共识"在 RL 里只能用于 trainer 组(rollout 之间的负载天然不同构)，ARGUS 的 kernel 级追踪无法覆盖 Ray RPC / HTTP 工具调用 / UCX 权重传输。跨角色、跨通信栈的统一 trace 仍然是空白**，这是你们如果想做原创贡献最清晰的方向。

### 3.3 Checkpoint

| 工作 | 出处 | 要点 |
|---|---|---|
| PHOENIX | arXiv:2607.01646(2026-07) | 零开销内存 checkpoint + 热替换:内存 checkpoint 完全离开关键路径、与计算重叠，无故障期零可测开销;配合 communicator 重建协议，用备机替换故障节点而不终止 job。512×A100、最大 65B，永久性节点故障后恢复 <40 秒 |
| ByteCheckpoint | NSDI'25，字节 | 生产级分布式 checkpoint，RobustRL 直接复用 |
| Gemini | SOSP'23 | 内存内 checkpoint + 分层恢复 |
| JIT Checkpointing | EuroSys'24，微软 | 故障发生瞬间才存，近零常态开销 |
| CheckFreq | FAST'21 | 自适应 checkpoint 频率 |
| Check-N-Run | NSDI'22，Meta | 差分 + 量化 checkpoint |
| Universal Checkpointing | 2024，DeepSpeed | 跨并行配置的 checkpoint 重切分 |
| 工程实现 | — | PyTorch DCP、Megatron dist-ckpt、DLRover Flash Checkpoint、`cuda-checkpoint` + CRIU |

> 504-GPU 报告的那条数据值得单独拿出来:重启加载只跑到峰值读带宽的 21.5%。多数团队以为 checkpoint 恢复受限于存储，实际受限于加载路径的并行度与元数据开销。优化恢复时间前先测这个比例。

### 3.4 弹性、冗余与重配置

| 工作 | 出处 | 要点 |
|---|---|---|
| SPARe | ICML 2026，arXiv:2603.00357 | 面向 100k+ GPU 的"重启主导"regime:在并行组之间堆叠冗余数据分片并自适应重排执行，从而在梯度同步阶段直接掩盖节点故障(不重启)。冗余度很高但计算开销仅 2–3×，优于复制方案的线性增长。600k GPU 下 time-to-train 降低 40–50% |
| Bamboo | NSDI'23 | 冗余计算抵抗抢占 |
| Oobleck | SOSP'23 | 预生成流水线模板做弹性恢复 |
| ReCycle | SOSP'24 | 用流水线固有冗余吸收故障 |
| Parcae | NSDI'24 | 抢占感知的主动弹性 |
| Varuna | EuroSys'22 | spot 实例上的大模型训练 |
| Tenplex / EasyScale / Unicron / Singularity | SOSP'24 / SC'23 / 2023 / 2022 | 状态重切分、确定性弹性、故障感知训练、透明抢占与迁移 |
| 工程实现 | — | TorchElastic、TorchFT、DLRover、nvidia-resiliency-ext |

### 3.5 网络与硬件

**HPN(SIGCOMM'24，阿里)、Meta RoCE 部署经验**(SIGCOMM'24)、TPU v4 + OCS 光交换(ISCA'23)、**Fire-Flyer(SC'24，DeepSeek)、NCCL communicator shrink API(支持非停机缩容)，以及上文 OrchestrRL 的 RFabric** (把光交换带进 RL 场景)。

### 3.6 推理服务可靠性(rollout 侧直接复用)

**DéjàVu(ICML'24)KV cache 流式与故障恢复;SpotServe(ASPLOS'24)spot 实例上的服务;Llumnix(OSDI'24)运行时请求迁移与重调度;EaaS**(2025)引擎即服务。
> Llumnix 的"请求级迁移"是 agentic rollout 最该借的能力:实例要下线时**迁移进行中的请求**而不是杀掉它。Belayer 的 KV 重建本质上是这一能力在"没法迁移"时的退化版本。

---

## 4. 工业界平台横向对比

| 组织 | 栈 | 可靠性关键词 |
|---|---|---|
| 字节 | MegaScale / ByteRobust / ByteCheckpoint / veRL | 全栈监控、job 级快速重启、热备池、并行组驱逐;20 万+ GPU;9600 卡三个月任务 ETTR 97% |
| Meta | Llama 训练栈 | ETTR 指标体系、故障表征公开、SDC 研究领先 |
| Google | TPU + OCS + Pathways | 光交换重构拓扑绕过故障;Gemini 训练 goodput 从 85% 提到 97% |
| 阿里 | RollArt / ROSE / Weave / Libra / C4 / HPN | 2026 年 agentic RL 基础设施产出最密集;3000+ GPU 生产验证;协同弹性(借 serving GPU) |
| 智谱 | slime + GLM-5 | 心跳驱动 rollout 容错 + router 注销;把容错/tracing/profiling/CI 当一等工程问题 |
| NVIDIA | DCGM / nvidia-resiliency-ext / NeMo | 硬件级健康检查与在训恢复库 |
| AWS | SageMaker HyperPod | 自动节点替换与作业续跑 |
| 蚂蚁 | DLRover | Flash Checkpoint、弹性训练、故障自愈 |
| 华为 | Ascend MindCluster / MindIO / MindSpeed RL | HBM-UCE 原地修复、进程级恢复、集群级故障管理;MindSpeed RL(arXiv:2507.19017)给出 Ascend 上的 RL 分布式数据流 |
| 微软 | SuperBench / SIGMA | 上线前主动验证;early-life 硬件的训练栈(2048 卡 75 天仅 1 次稳定性事件) |

---

## 5. 2026 相对 2025 发生了什么变化

1. 主战场从预训练迁到 RL / Agentic RL 后训练。 2025 年这个方向只有 RobustRL、Laminar、StreamRL 少数几篇;2026 年 OSDI 直接收了 Weave、RLinf，arXiv 上 Belayer、Libra、ROSE、BiDiRL、OrchestrRL、RollArt、RFT-FM 密集出现。
2. **容错粒度继续下沉:job 级(ByteRobust)→ 角色级(RobustRL)→ 请求/轨迹级**(Belayer、RollArt)。
3. **"环境"成为一等公民**。Belayer 给环境做 checkpoint-restore，Rollout Infrastructure Tax 直接量化沙箱基座的成本差(110× 冷启动、1.8× 总 worker-hours)。
4. **故障边界从系统扩展到算法**。RFT-FaultBench 把"训练跑飞"纳入故障管理闭环。
5. fail-slow 压过 fail-stop 成为主要损耗。ARGUS 的 59% / 34.59% 与 Guard 的方差 20%→1% 说明:宕机好歹看得见，慢节点吃掉的算力更多且更隐蔽。
6. **常开可观测性变得可行**。ARGUS 把 always-on 细粒度 tracing 的开销压到 <2%、数据量压 3700×，"平时不开 profiler"这个借口正在消失。
7. **不重启成为新目标**。SPARe 在梯度同步中直接掩盖故障，PHOENIX 热替换 <40s 且不终止 job。
8. 弹性从"申请新资源"变成"借用已有资源"。ROSE 借 serving 集群、Weave 借另一个任务的气泡、Libra 借对侧阶段的 worker、RobustRL 借 rollout 机器——同一条思路的四种实现。
9. **国产/新硬件成为显式议题**。Libra 在 160×Ascend 910B3 上验证，SIGMA 专门面向 early-life 加速器。

---

## 6. 尚未解决的空白(也是可发力点)

1. **跨角色、跨通信栈的统一追踪**。NCCL Flight Recorder 管不到 Ray RPC、HTTP 工具调用、UCX 权重传输、沙箱 RPC。RL 的关键路径穿过至少四种通信栈，目前没有任何一个工具能给出端到端因果链。
2. 非同构副本的 fail-slow 检测。SCOUT / Minder 依赖"等价副本"假设，而 rollout 实例因为轨迹长度差异天然不同构，需要新的基线建模(例如按 token 归一化的 TPS 而非绝对 TPS)。
3. **管理面/数据面角色的高可用**。几乎所有工作都在保 trainer 和 rollout，而 Driver、Store、Router/LoadBalancer 这些单点在论文里基本不出现——但在真实系统里它们的故障率并不低。
4. 在线 SDC 检测在 RL 中的形态。SCOUT 用共识法检测 SDC，但 rollout 是采样式的、本身带随机性，如何区分"采样差异"与"硬件静默错误"没有答案。
5. 轨迹内权重版本一致性的系统化保证。异步 RL 中一条轨迹可能前缀用 W_k、后缀用 W_{k+1}，重要性采样比失真。Belayer 在环境侧提出了 prefix 一致性约束，但权重版本维度还没有系统工作。
6. 非 NVIDIA 硬件的通信库级追踪。HCCL 没有 Flight Recorder 对等物，Ascend 上的 hang 定位仍以日志和栈为主。
7. **可靠性与算法效果的联合指标**。ETTR 高但 reward 跑飞 = 更大的浪费。RFT-FM 开了个头，但和系统侧 ETTR 尚未统一。

---

## 7. 映射到你们的 AgenticRL 平台(verl + agent-lightning，Ascend + K8s)

恢复动作之间的层次与升级关系见 §2.1。本节按模块逐条给出最值得参考的来源:

| 你们的模块 | 最相关的工作 | 具体可搬运的东西 |
|---|---|---|
| 故障检测(Daemon) | RobustRL 角色+阶段感知;slime 心跳;ARGUS 渐进式诊断 | 阈值按角色和阶段分别定义;心跳走 `/health_generate` 而不是靠 GPU 利用率;首次心跳要有 300s+ 的宽限(CANN 图编译) |
| LLMProxy / LoadBalancer | GLM-5 + slime | 已具备:`6.3` 第 2 步「LoadBalancer 移除故障实例」+ LLMProxy 按 LiteLLM 策略重试其他后端，与 slime 的 router 注销等价。可补的是 slime 的 `first-wait 300s` 宽限(CANN 图编译首启慢) |
| TokenRetry ↔ InstanceRestart 之间 | Belayer | 补一档"保留权重与 KV arena、只重建请求态"的恢复，官方数据比冷启动快 42× |
| 环境 / 工具 / 沙箱故障 | Belayer、Rollout Infrastructure Tax、LiteResearcher、RFT TE 族 | 当作「结果」管，未当作「故障对象」管。已有:Daemon 层「标记 failed 只丢弃该样本」+ Agent Runner 层重试/放弃 + ErrorClassifier 异常分类 + 失败率熔断。缺的是:(a) `4.1` 九个故障对象全是基础设施，无「环境/工具」行，导致 `4.2` 没地方挂差异化处置;(b) ErrorClassifier 的分类只用于决策分支，未上报为按根因分解的指标(沙箱冷启动超时 / OOM / 外部限流 / 格式损坏无法分开统计);(c) 处置只有「丢弃」一档，成本随轨迹长度线性增长(Belayer 实测轨迹中位数 460.8s，1% 环境动作失败 → rollout +48.7%，上 checkpoint-restore 后 → 1.5%);(d) 丢弃非随机——长轨迹/难任务更易撞故障，batch 里系统性少掉难样本(即 RFT 的 TE-3)，且在 ETTR 上完全不可见;(e) 失败率阈值全局不分来源，沙箱镜像坏 / NPU 坏 / 外部限流触发同一动作。最小改动:4.1 加一行「Agent 环境/工具调用」→ ErrorClassifier 结果上报为指标 → 丢弃率与轨迹长度联合上报，再决定是否值得付 CRIU 的代价 |
| Store + 轨迹消费状态机 | RobustRL §5.2.2 RequestManager;Belayer prefix 一致性 | 按工具轮次持久化;恢复时取"两边更保守的那个"状态;幂等键 (rollout_id, global_step) |
| 权重同步 | RobustRL relay;SparseRL-Sync;ROSE | 实例数弹性时不要建固定 HCCL 通信域;稀疏同步把传输窗口缩两个数量级，等价于降低被打断概率 |
| Checkpoint | PHOENIX;ByteCheckpoint;504-GPU 报告 | per-step 异步 checkpoint 作为主路径(临终保存降级为 P1);先测一下恢复时用到了峰值读带宽的百分之几 |
| 借机器 / 弹性 | RobustRL §5.1.3;Libra;ROSE;Weave | K8s 原生做法:让 rollout Pod spec/镜像与训练 worker 完全一致，故障时秒级顶替 |
| 慢节点 | ARGUS;Guard;Minder | 按角色分组、对同角色中位数做比较;上线前 node-sweep(Guard 把 step 方差从 20% 打到 1%) |
| 节点黑名单 | 504-GPU 报告 | top-3 节点贡献 50%+ 的排除次数 → 黑名单/taint 的投入产出比极高 |
| 自动重试策略 | 504-GPU 报告 | 自动重试成功率 33.3%，是人工的 2.7×;重试间隔中位数 11 分钟可作为你们退避策略的起点 |
| 训练跑飞 | RFT-FM / RFT-FaultBench | 在告警面板里并列系统故障与训练动力学异常 |
| 国产卡语境 | Libra(160×910B3)、MindSpeed RL、SIGMA、MindIO | Libra 是少见的同时在 A800 和 910B3 上给出数据的 RL 系统工作，可直接对标 |

### 7.1 三方对比:你们的方案 vs RobustRL vs Belayer

> 已按原文核对更正三处:(a) RobustRL 的故障模型**是硬件的**——其论文 §2.2 以机器故障立论，点名 GPU 掉卡与 ECC，非早期表述的"换机器";(b) 你们**已有**环境/工具故障的兜底链;(c) 你们**已有**路由摘除。

| 维度 | RobustRL | Belayer | 你们的 AgenticRL |
|---|---|---|---|
| 故障模型 | 硬件为主——机器故障(GPU / 网络 / CPU / 内存 / 磁盘)，点名掉卡 + ECC | 软件为主——Python 异常、进程崩溃、worker hang、容器崩溃 | 硬件 + 基础设施——NPU 掉卡、HBM-UCE、Pod / 节点 / 集群 |
| 明确不管 | SDC、straggler(自陈"架构便于集成"，即未做) | trainer、SDC、fail-slow、硬件故障、权重/KV server、容器外状态 | fail-slow、SDC;环境故障无独立故障对象 |
| 容错粒度 | 角色级:trainer / rollout / 管理角色 | 请求级 + 环境级 | 六档:Token → 实例 → Step → 进程 → 作业 → 编排 |
| 判死机制 | trainer:TensorCore=0 持续 5min(仅训练阶段);rollout:TPS=0 持续 60s + 心跳确认 | shadow 监控主 worker 在调度循环刷新的临时文件 mtime;接管前查 GPU 健康 + owner 存活 | health_check 心跳;TPS=0 / 60s + 心跳(与 RobustRL 一致);ErrorClassifier 关键词 + 异常链遍历 |
| 判死权威 | RolloutManager / ElasticPolicy;硬件根因诊断外包给平台 | router 是唯一真相源——shadow 必须先让它注销旧 worker 再注册自己 | Driver/SDK + LLMServerManager + MindIO 三方均可发起，仲裁未定义 |
| trainer 恢复 | 全体 trainer 一起重启 + 逐 step ByteCheckpoint(阻塞仅 GPU→内存 ~3s，落盘异步) | 不做，沿用预训练异步 ckpt | StepRetry:MindIO 挂起进程、不杀进程不重建资源、原地修复 HBM-UCE |
| rollout 恢复 | 重启 worker + 远程重载权重(冷启动) | shadow 接管 ~1s，保权重 + KV arena，仅 1.84GB 显存;单次故障训练时间 +1.16% vs 基线 +8.72% | `InstanceRestart` 冷启动 + CheckpointEngine 推权重 |
| 请求态保全 | RequestManager 按工具轮次保留轨迹;故障时重分配给存活 rollout | 每 256 token 增量记 prefix，切换后重放重建该请求 KV | 五层兜底:LB 重推 → LLMProxy 重试 → Runner 重试/放弃 → Daemon 超时丢样本 |
| 路由摘除 | RolloutManager 把先前结果重分配 | router 注销旧 / 注册新 | 已有:LoadBalancer 移除故障实例 + 新实例加回 |
| 环境/工具 | 沙箱移出故障域(是保住轨迹的前提而非解法);AgentWorker 属管理角色，挂掉 = 全任务重启 | overlay 可写层 + CRIU，冻结在 action 边界、原子发布;自适应策略藏掉 96.8% 开销 → rollout +1.5% vs 重启基线 +48.7% | 仅「丢弃整条轨迹」一档;损失单位是轨迹 vs Belayer 的一个 action，放大约 20× |
| 权重同步容错 | UCX + relay 中继，中继故障可换中继续传;235B/470GB ~6s | 假设权重/KV server 存活;shadow 反复失败则标记 GPU 不健康并排除 | 未见设计 |
| 硬件故障 | 换机器(ECC / 掉卡);§5.1.3 借 rollout 机器秒级顶替，免等 gang scheduling | 明确不保证 warm recovery(CUDA context 损坏 / driver reset / GPU 硬件故障) | 三级处置:MindIO 原地修(秒级)→ 池内换健康节点(~1–2min)→ PodReschedule |
| 编排层 | 无 | 无——| 独有:StoreRestart / JobResubmit / PodReschedule / ClusterRecreate + CRD / Controller / KubeRay |
| 验证方式 | 仅注入 kill all trainers;硬件路径与借机器路径均未验证 | Pumba 注容器 fail-stop + `sitecustomize.py`/`LD_PRELOAD` 注软件故障;600 次试验 593 次中断活跃执行，全部恢复到 golden final state | 生产运行，未见系统性故障注入 |
| 验证规模 | 256×H20(端到端 128 卡)，最大 235B-A22B | 32×H200(4 节点)，最大 32B | 生产集群 |

#### 方案可改进之处

下列各项按性质分组，每项给出依据与建议动作。优先级排序见末尾。

**一、需先行验证的前提**

| 事项 | 问题 | 依据 |
|---|---|---|
| MindIO 在 PyTorch 路径的可用性 | TFT 的特性清单、环境变量(`MS_ENABLE_TFT`)、`TFTRegister` callback 与 `msrun` 启动方式均来自 MindSpore / MindSpore Transformers 文档;昇腾对 PyTorch 的对接文档是另一份(对接 MindSpeed-LLM)。verl + FSDP 既非 MindSpore 亦非 Megatron 系 | 需确认三点:FSDP 的参数分片布局能否满足 TTP/UCE/ARF 所要求的「卡间副本关系」;「UCE 仅支持图模式」在 torch_npu 下的对应约束;`sink_size=1` 是否有等价限制 |

此项影响 `StepRetry` 一档的成立性。该档是当前方案相对 RobustRL 与 Belayer 的主要优势(前者全体 trainer 重启，后者不处理 trainer 故障)，其前提若不成立，该优势随之失效。建议优先验证。

**二、结构性问题(影响恢复正确性)**

| 事项 | 问题 | 建议 |
|---|---|---|
| 判死权威未收敛 | Driver/SDK、LLMServerManager、MindIO 三方均可发起故障判定，仲裁规则未定义，存在重复判死与脑裂风险 | 编排层已有正确范式——ClusterX 只在 AgenticRLJob 上打故障决策标注、不给 RayJob 打标注，且 Controller 在恢复期间停止调谐。将同一范式下沉至进程内层 |
| SDK 未定义 | 该名称出现九次，承担进程内全部恢复决策权，但不在 §2 组件清单与架构图中，无接口、无归属。§6.6 将 `JobRestart` 的决策方记为「Driver / SDK」，而该档的触发条件是 Driver 崩溃，决策方在字面上自指 | 显式定义为具名组件(如 `RecoveryController`)，明确进程归属、上报接口与仲裁规则。前四档可由进程内决策器承担;`JobRestart` 的决策方必须位于 Driver 进程之外。Ray 的架构决定 Driver 无法自救，进程外看门狗是架构强制项而非可选项 |
| 重试缺乏幂等键 | LLMProxy 与 LoadBalancer 串联且各自重试，叠加后单请求最多触发 N×M 次，且在实例刚失效、余量实例承接溢出流量时放大。若请求实际成功而响应丢失，重试将产生重复 Span | 在 Span 写入侧引入幂等键(如 `rollout_id`、`turn_idx`、请求唯一 id)。消费侧的轨迹状态机无法识别不同 span_id 的重复记录，防重必须置于写入侧。建议同时将重试收敛到单层:LoadBalancer 承担数据面重试，LLMProxy 作为无重试的旁路 |
| 控制面组件不在故障对象表内 | §4.1 的九个故障对象均为基础设施。LLMProxy 位于数据链路(Span 导出为训练数据唯一来源)，LoadBalancer 是 rollout 链路必经单点并承担心跳摘除，二者均无对应恢复档 | 补入故障对象表并给出恢复档。二者为无状态服务，Deployment 多副本 + Service 即可，成本低，但需先纳入表中 |
| 检测器自身故障 | 承担故障检测的组件失效后，系统将失去发现其他故障的能力，表现为静默停滞而非报错。RobustRL 存在同类问题(RolloutManager 既是被保护对象又是 TPS 采集源) | 检测链需终止于不依赖被检测系统的外部机制，如 K8s liveness probe 或 ClusterX |

**三、效率问题(影响 MTTR 与 ETTR)**

| 事项 | 问题 | 建议 |
|---|---|---|
| `InstanceRestart` 为冷启动 | 需重做权重加载、KV arena 分配与图编译。Belayer 基于 58 个真实推理引擎 bug 的实测表明，fail-stop 情形下权重与 KV arena 有 82.8% 可复用，完整 KV 镜像仅 15.5% 可复用;据此采用「保留权重与 arena、丢弃 KV 内容并由 prefix 重放重建」的切分，恢复时间约 1 秒，较冷启动快 42× | 引入 shadow worker，将权重与 KV arena 的所有权从 worker 进程剥离。**跨进程设备内存共享在昇腾已可行**:CANN 提供 `aclrtIpcMemGetExportKey` / `aclrtIpcMemImportByKey` / `aclrtIpcMemSetImportPid`，以及物理内存路线 `aclrtMallocPhysical` + `aclrtMemExportToShareableHandle`(对应 CUDA VMM);已有两处先例——warmload 实现跨进程权重共享(消费者张量为另一进程内存的视图，**生命周期依赖加载器存活**，正是 shadow 所需性质)，vLLM-Ascend PR #15832 用 torch-npu IPC handle 跨进程共享 KV cache 分配。**真正的未知数是图侧**:已编译的 GE / ACL 图能否在另一进程重映射设备地址，无证据,而图编译恰是收益最大的一项，spike 重点应放在此。另两项风险:跨进程 NPU Event 观察不可靠(该 PR 实测，最终将 Event 全部保留在 Worker 内);容器内跨进程共享需共享 IPC namespace 与正确设备挂载，未见验证。CANN 图编译使昇腾冷启动成本高于 H200，该项收益应高于论文报告值 |
| 恢复阶梯缺少根因驱动的跳级 | 阶梯为串行结构，每档须先失败方可进入下一档。但部分根因在决策前即可判定低档必然失败:NPU 掉卡时设备已不可见，`StepRetry` 依赖的 MindIO 原地修复无从执行;池内无空闲健康节点时，`ProcessRestart` 降级至 `JobRestart` 仍在同一 RayCluster 内，同样缺节点 | 增加前置判定:根因为掉卡则跳过 `StepRetry`;池内无空闲健康节点则直接进入 `PodReschedule`。两项均为决策前可查的廉价事实(MindIO 故障码、RayCluster 可用资源)，无需试错 |
| 升级判据不可判定 | 现有条件为「多次 InstanceRestart 仍无法恢复」「若 ProcessRestart 失败」等定性描述，无判定点，无法实现为程序分支。仅 `storeUnavailableCount` 与 `backoffLimit` 可计数 | 采用 RobustRL 的可复现性判据形式:首迭代或恢复后首步异常、同一 step 第二次失败、同一恢复动作第二次失败。三者均指向「故障可复现，故非硬件故障，角色级重启无效」，且可直接实现为条件判断 |
| `StoreRestart` 影响半径与其位置不符 | 流程第 3 步为「删除 RayJob」，实际包含一次 `JobResubmit`，影响半径不小于后者，而其在阶梯中排序更靠前。该流程对数据库型 Store 同样执行，但此类 Store 自带持久化，Pod 重启不致数据丢失 | 采用有界本地缓冲加背压:写入方缓冲一个有界窗口，窗口满则阻塞上游，Store 恢复后排空;仅当停机超出窗口才升级至删除 RayJob。该方案同样保证数据完整性(数据被持有而非写入失效端点)，但避免将秒级故障放大为分钟级重建。建议数据库型与内存型 Store 采用不同流程 |
| 节点隔离未定义具体动作 | `ProcessRestart` 阶段一称「故障确认与隔离」，未说明隔离的实现。若不施加 taint 且无跨作业黑名单，调度器可能在 `PodReschedule` 时将新 Pod 调回故障节点 | 明确隔离为「打 NoSchedule taint 并写入跨作业持久化黑名单」。504-GPU 集群运维报告显示 63 个节点中前 3 个贡献超过 50% 的排除次数，故障节点高度集中。注意 UCE 修复成功的节点应视为健康、不进黑名单，两类硬件故障的节点处置相反 |

**四、覆盖缺口**

| 事项 | 问题 | 建议 |
|---|---|---|
| 环境/工具故障未建模为独立故障对象 | §4.1 九个故障对象均为基础设施。现有处置(Agent Runner 重试或放弃、Daemon 超时丢弃样本、ErrorClassifier 分类、失败率熔断)寄生于 `6.3 InstanceRestart` 的兜底链;当沙箱失效而推理实例健康时，该链不被触发。处置仅「丢弃」一档，损失单位为整条轨迹，而 Belayer 的损失单位为单个 action;按轨迹时长中位数 460.8 秒、SWE 任务至多 20 轮计，差异约 20 倍。丢弃在 ETTR 上不可观测，且因长轨迹与难任务遭遇环境故障概率更高而呈非随机分布，系统性削减难样本 | 分三步推进:(1) 补入故障对象表，并将 ErrorClassifier 的分类结果上报为按根因分解的指标;(2) 以「被丢弃轨迹平均长度 / 成功轨迹平均长度」为看板指标，比值显著大于 1 表明样本分布已偏;(3) 确认 Agent Runner 环境是否容器内自包含(存在 host 挂载卷或写外部服务则 Belayer 的 prefix 一致性保证不适用)，再决定是否引入环境 checkpoint;引入时建议先实现文件系统层归档，暂不引入 CRIU 进程快照 |
| fail-slow 与 SDC 未覆盖 | 三方均未覆盖，属该领域共同空白，非本方案特有短板 | 优先级可低于上述各项。若投入，fail-slow 的可行切入点是按角色分组、以同角色中位数为基线(rollout 实例因轨迹长度差异天然非同构，需以按 token 归一化的吞吐替代绝对吞吐) |

**五、验证方法**

现有设计未见系统性故障注入。作为对照，RobustRL 采用每若干步终止全部 trainer 进程的方式注入(其硬件路径与借机器路径因此未获验证);Belayer 使用 Pumba 注入容器 fail-stop，并以 `sitecustomize.py` 与 `LD_PRELOAD` 注入软件故障，以 600 次任意时刻试验验证恢复正确性。建议建立最小注入集合:终止 Driver 进程、删除 Store Pod、删除 head Pod、NPU 拔卡、HBM-UCE 注入、网络分区、NFS 挂起、单实例人为降速，并以「恢复后训练进度与轨迹消费状态是否与预期一致」为验收判据。

**优先级建议**

1. MindIO 在 PyTorch 路径的可用性验证(决定 `StepRetry` 是否成立)
2. SDK 具名化并拆分为进程内决策器与进程外看门狗(决定 `JobRestart` 是否自洽)
3. 判死权威收敛、重试幂等键(正确性)
4. 根因驱动跳级、升级判据可判定化(低成本、直接降低 MTTR)
5. 环境/工具故障的量测(先量测，后决定是否投入机制)
6. ACL 跨进程内存句柄 spike 与 shadow worker(收益最大，门槛最高)
7. 故障注入验证集合
**故障模型覆盖矩阵**:

| | fail-stop | fail-stall(hang) | fail-slow | Byzantine(SDC) |
|---|---|---|---|---|
| RobustRL | 有 | 超时转化 | 自陈未做 | 自陈未做 |
| Belayer | 有 | 心跳转化 | 明确排除 | 明确排除 |
| 你们 | 有 | 超时转化 | 无 | 无——|

> 三家都只打左边两档。fail-slow 与 SDC 在 RL 后训练是全行业空白，不是你们的短板。注意 hang 严格不属于 fail-stop，三者都是用超时/心跳把 fail-stall 转化成 fail-stop——而在 agentic RL 里「无进展」是正常状态(等工具返回)，故阈值必须按角色与阶段分别定义。

**结论**:你们**两头最强、中间差一格**。

- trainer 侧**领先两篇论文**(StepRetry 原地修复 > RobustRL 全体重启 > Belayer 不管);编排层与硬件故障为你们独有。
- 缺口只剩两条:① `InstanceRestart` 是冷启动;② 环境故障只有「丢弃」一档。
- 新增风险:**判死权威不唯一**。Belayer 收归 router、RobustRL 收归 RolloutManager，而你们 Driver/SDK、LLMServerManager、MindIO 三方都能发起，需确认不会同时判死同一对象(脑裂)。
## 8. 参考文献清单

### 2026 新工作(本次已核对原文)

- Belayer: Efficient Fault Tolerance for LLM Agentic RL Training — arXiv:2608.14635 — https://arxiv.org/abs/2608.14635
- Towards Robust LLM Post-Training: Automatic Failure Management for Reinforcement Fine-Tuning (RFT-FM / RFT-FaultBench) — arXiv:2605.04431 — https://arxiv.org/abs/2605.04431
- RollArt: Disaggregated Multi-Task Agentic RL Training at Scale — arXiv:2512.22560 — https://arxiv.org/abs/2512.22560
- Libra: Efficient Resource Management for Agentic RL Post-Training — arXiv:2606.03077 — https://arxiv.org/abs/2606.03077
- Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training — OSDI '26 — https://www.usenix.org/system/files/osdi26-wu-tianyuan.pdf
- RLinf: Flexible and Efficient Large-Scale RL via Macro-to-Micro Flow Transformation — OSDI '26 — https://www.usenix.org/conference/osdi26/technical-sessions
- ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL — arXiv:2605.06534 — https://arxiv.org/abs/2605.06534
- BiDiRL: Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training — arXiv:2607.09207 — https://arxiv.org/abs/2607.09207
- OrchestrRL: Dynamic Compute and Network Orchestration for Disaggregated RL — arXiv:2601.01209 — https://arxiv.org/abs/2601.01209
- The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning — arXiv:2607.01415 — https://arxiv.org/abs/2607.01415
- SparseRL-Sync: Lossless Weight Synchronization with ~100× Less Communication — arXiv:2605.07330 — https://arxiv.org/abs/2605.07330
- AReaL-DTE: Sparse Policy-Weight Transfer for Online Agentic RL — arXiv:2608.00455 — https://arxiv.org/abs/2608.00455
- ARGUS: Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU Clusters — arXiv:2606.20374 — https://arxiv.org/abs/2606.20374
- SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training — arXiv:2608.11034 — https://arxiv.org/abs/2608.11034 — 代码 https://github.com/LMResiliency/lm-resiliency
- Guard: Scalable Straggler Detection and Node Health Management for Large-Scale Training — MLSys 2026，arXiv:2605.17879 — https://arxiv.org/abs/2605.17879
- SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining with 100k+ GPUs — ICML 2026，arXiv:2603.00357 — https://arxiv.org/abs/2603.00357
- PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint — arXiv:2607.01646 — https://arxiv.org/abs/2607.01646
- From Detection to Recovery: Operational Analysis on LLM Pre-training with 504 GPUs — arXiv:2605.09370 — https://arxiv.org/abs/2605.09370
- SIGMA: An AI-Empowered Training Stack on Early-Life Hardware — arXiv:2512.13488 — https://arxiv.org/abs/2512.13488
- GLM-5: from Vibe Coding to Agentic Engineering — arXiv:2602.15763 — https://arxiv.org/abs/2602.15763
- **slime 容错文档** — https://github.com/THUDM/slime/blob/main/docs/zh/advanced/fault-tolerance.md
- Orchard: An Open-Source Agentic Modeling Framework — arXiv:2605.15040
- LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent — arXiv:2604.17931
- EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL — arXiv:2605.18703
- MindSpeed RL: Distributed Dataflow for Scalable and Efficient RL Training on Ascend NPU Cluster — arXiv:2507.19017
- Laminar: A Scalable Asynchronous RL Post-Training Framework — **EuroSys 2026**，arXiv:2510.12633 — https://arxiv.org/abs/2510.12633 — https://dl.acm.org/doi/10.1145/3767295.3803580
- Unleashing Efficient Asynchronous RL Post-Training via Staleness-Constrained Rollout Coordination — arXiv:2601.12784
- Periodic Asynchrony: An On-Policy Approach for Accelerating LLM Reinforcement Learning — arXiv:2511.18871

### 2025 及更早(来自既有整理，引用前请核对)

- RobustRL — arXiv:2512.22492(已核对作者与日期，**会议归属未证实**)
- RLBoost — arXiv:2510.19225;StreamRL — 2025;AReaL — NeurIPS 2025;AReaL-Hex — arXiv:2511.00796
- ByteRobust(2025)、MegaScale(NSDI'24)、ByteCheckpoint(NSDI'25)、C4(SIGCOMM'24)
- Minder(NSDI'25)、Mycroft(2025)、Greyhound/FALCON(ATC'25)、SuperBench(ATC'24)、TRANSOM(2023)、L4(arXiv:2503.20263)
- TrainCheck(OSDI'25)、TrainVerify(SOSP'25)、What-if analysis(OSDI'25)
- Gemini(SOSP'23)、JIT Checkpointing(EuroSys'24)、CheckFreq(FAST'21)、Check-N-Run(NSDI'22)、Universal Checkpointing(2024)
- Bamboo(NSDI'23)、Oobleck(SOSP'23)、ReCycle(SOSP'24)、Parcae(NSDI'24)、Varuna(EuroSys'22)、Tenplex(SOSP'24)、EasyScale(SC'23)、Unicron(2023)、Singularity(2022)
- Revisiting Reliability in Large-Scale ML Research Clusters(HPCA'25，arXiv:2410.21680)、Acme(NSDI'24)、Philly(ATC'19)、Llama 3 训练报告(2024)
- HPN(SIGCOMM'24)、Meta RoCE(SIGCOMM'24)、TPU v4(ISCA'23)、Fire-Flyer(SC'24)
- DéjàVu(ICML'24)、SpotServe(ASPLOS'24)、Llumnix(OSDI'24)
- 工具与开源:DCGM、NCCL Flight Recorder、nvidia-resiliency-ext、TorchElastic/TorchFT、DLRover、MindIO/MindCluster
