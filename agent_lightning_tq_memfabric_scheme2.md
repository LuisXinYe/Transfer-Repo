# TQ 接入 MemFabric：方案 2（保留 SimpleStorage，NPU worker 直读 HBM）

2026-11-05 · 适用于 `feature/chartqa-multimodal-tq-storewrite`（verl 0.9.0、TransferQueue 0.1.8、Atlas 800T A3 双机）

标注约定：【已核实】读过源码或实测过；【推断】按机制推出、未实测；【待验证】方案能否成立取决于它，需要先做实验。

---

## 1. 要解决什么

训练关键路径上真正要付的，是"读取一跳"：worker 从存储单元拉取一个阶段（old_log_prob、ref、actor 更新）要用的数据。

### 1.1 现状：worker 跨机读一个张量字段（如 `input_ids`、`responses`）

```
【机器 A：存储单元所在节点】

  存储单元进程的内存（每行一个张量）
      │
      │ ① send()：CPU 把字节拷进内核                    ← CPU 拷贝
      ▼
  A 的内核 socket 缓冲区
      │
      │ ② A 的网卡用 DMA 从内存读出，发上网线           ← 网卡硬件
      ▼
════════════ 网线（主机网卡，普通 TCP） ════════════
      │
      │ ③ B 的网卡用 DMA 把包写进内存                   ← 网卡硬件
      ▼
【机器 B：训练 worker 所在节点】

  B 的内核 socket 缓冲区
      │
      │ ④ recv()：CPU 把字节拷到用户内存                ← CPU 拷贝
      ▼
  worker 进程的主机内存（一行一块，零散）
      │
      │ ⑤ 拼批 _pack_field_values：拼成连续的 jagged 张量 ← CPU 拷贝
      ▼
  worker 进程的主机内存（连续）
      │
      │ ⑥ H2D：NPU 的 DMA 引擎拷上卡                    ← NPU DMA
      ▼
  worker 本卡 HBM → 计算
```

| 步骤 | 机器 | 从 → 到 | 谁搬字节 | 依据 |
|---|---|---|---|---|
| ① | A | 存储单元内存 → A 内核缓冲区 | CPU（`send`） | 【推断】Linux TCP 常规行为 |
| ② | A | A 内核缓冲区 → 网线 | A 的网卡 DMA | 【推断】同上 |
| ③ | B | 网线 → B 内核缓冲区 | B 的网卡 DMA | 【推断】同上 |
| ④ | B | B 内核缓冲区 → worker 主机内存 | CPU（`recv`） | 【推断】同上 |
| ⑤ | B | 零散的行 → 一块连续张量 | CPU | 【已核实】`simple_storage_manager.py:350-366, 419` |
| ⑥ | B | 主机内存 → HBM | NPU 的 DMA 引擎 | 【推断】 |

补充三点：

- ①–④ 之外，CPU 还要跑两端的 TCP/IP 协议栈：分段、确认、重传、处理中断。这些不是字节拷贝，但同样占 CPU。
- **同机时**没有 ② 和 ③：TCP 走回环，不经过网卡，内核直接把数据从发送缓冲区挪到接收队列。①④⑤⑥ 照旧。
- **像素**现在放在 `multi_modal_inputs` 的 dict 里，跳过 ⑤，用户态不拷贝【已核实】。但 verl 在上卡前可能会自己 `torch.cat` 一次【推断】。

一共六步：3 次 CPU 拷贝、2 次网卡 DMA、1 次 H2D。3 次 CPU 拷贝里：① 和 ④ 是内核态拷贝，⑤ 是 TQ 客户端在用户态拼批。像素不经过 ⑤，只有 ① 和 ④ 两次。

### 1.2 方案 2：同一次读取

```
【机器 A】

  存储单元贡献给 MemFabric 池的 DRAM 段（写入时已经放好）
      │
      │ ① GH2L：一次 DMA，由 B 上的 worker 发起，CPU 不碰数据字节
      │    · 同一超节点：device_sdma，走 HCCS
      │    · 否则：device_rdma，走 NPU 自带的 RoCE 网口
      ▼
【机器 B】

  worker 本卡 HBM（按偏移直接排成 jagged 布局）→ 计算
```

开始前，worker 要先用 ZMQ 向存储单元要一次地址。这是一条小消息，不带数据。

六步缩成一步：① 到 ④ 的拷贝和协议栈、⑤ 的拼批、⑥ 的 H2D 都不需要了。注意走的也不再是主机网卡：HCCS 是超节点内的专用互联，RoCE 网口在 NPU 上，和主机网卡是不同的物理链路。

### 1.3 为什么快路径只能是"读进 NPU"

在 MemFabric 的 SDMA 实现里，不管池段是 HBM 还是 DRAM，只要目标是本机 HBM（拷贝类型 `G2L` / `GH2L`），都直接调 `CopyG2G`，一次 DMA 完成。反过来，只要源或目标是本机主机内存（内部函数 `CopyLH2GD`、`CopyGD2LH`、`CopyLH2GH`、`CopyGH2LH`），都要先 `AclrtMalloc` 一块临时 HBM 中转（`hybm_data_op_sdma.cpp:106-174, 185-230, 373`）【已核实】。所以快路径只能是"读进 NPU"。

**收益上限**：等于现在读取一跳的实际耗时，这个数**还不知道**。在 TQ 分支设 `VERL_LOGGING_LEVEL=INFO` 跑几步，看 tqbridge 打出的 `Task ... is getting len_samples=..., cost time:` 就能拿到。

---

## 2. 原先的 TQ 方案（现状）

```
rollout runner ──> agentlightning store 进程
                     │ MultimodalRowPool 建行（spawn 进程池，经 /dev/shm 交回）
                     │ async_kv_batch_put
                     ▼
              SimpleStorage 客户端：msgpack 编码，张量作为独立帧（零拷贝）
                     │ ZMQ tcp://，send_multipart(copy=False)
                     ▼
              SimpleStorageUnit × 2（global_index % 2 路由）
                StorageUnitData: field → {global_index → 张量视图（引用收到的 ZMQ 帧）}

driver ── KVBatchMeta（只有 key 和 tag）── Ray ──> 训练 worker

训练 worker（verl tqbridge）
  async_get_data ──ZMQ──> 存储单元回数据帧 ──> decode（frombuffer，零拷贝）
                 ──> _pack_field_values（拼成 jagged 嵌套张量，CPU 拷贝）──> TensorDict（CPU）──> H2D
  async_put(log_probs, entropy) ──ZMQ──> 存储单元

driver（优势计算等）：取回小字段 → 计算 → 写回，全走 ZMQ
```

要点：

- 写入方是 store 进程，读取方是 worker 和 driver，三者都通过 ZMQ 和存储单元通信。控制和数据走的是同一条 ZMQ 连接【已核实】。
- 编码时张量不拷贝（`serial_utils.py:200-213`）；`recv_multipart(copy=False)` 后用 `frombuffer` 建视图（`simple_storage.py:289`，`serial_utils.py:297-309`）【已核实】。
- 拼批时有一次 CPU 拷贝：`torch.nested.as_nested_tensor(..., layout=jagged)`（`simple_storage_manager.py:350-366, 419`）【已核实】。
- 像素放在每行一个 dict 里（`multi_modal_inputs`，`tq_trajectory.py:318`），走 `NonTensorStack`，用户态不拷贝【已核实】。
- 每跳 TCP 在两端各有一次内核拷贝，同机走回环也一样【推断】。

---

## 3. 方案 2 整体框架

### 3.1 一句话

**控制面不动，数据面分成两路**：

- NPU worker 读大张量字段时走快路径：存储单元只回"数据在池里的地址"，worker 自己用 MemFabric 把数据从池里拷进本卡 HBM。
- 其余全部照旧走 ZMQ：store 写入、driver 读写小字段、非张量字段。

### 3.2 组件

| 组件 | 角色 | 变化 |
|---|---|---|
| TQ 控制器 | 元数据、分区、`KVBatchMeta` | 不变 |
| SimpleStorageUnit | 存数据 | **加入 MemFabric**，向池贡献一块内存段；张量字节放进这块段里；维护目录 `(字段, global_index) → (GVA 偏移, 字节数, dtype, 形状)`（GVA：全局虚拟地址，MemFabric 池里统一编址的地址） |
| `PayloadTransfer` 的 MemFabric 实现 | 决定一次 put/get 的数据怎么传 | **新增**。接口来自上游 `rdma` 分支：`put` / `get` / `handle_request` / `bootstrap_info`（`payload_transfer/base.py`）【已核实】 |
| SimpleStorage 客户端（`get_data`） | 发请求、收数据、拼批 | **新增快路径**：收到地址后发起"池 → 本卡 HBM"的拷贝（DRAM 段用 `GH2L`，HBM 段用 `G2L`），在卡上拼批 |
| verl tqbridge | worker 侧取数、写回 | **改**：取数时告诉客户端目标设备是本卡，返回的 TensorDict 留在 NPU 上 |
| store、driver | CPU 进程 | 不接入 MemFabric，继续走 ZMQ |

### 3.3 写入路径（store → 存储单元）

1. store 照旧通过 ZMQ 把编码后的帧发给存储单元。传输方式不变。
2. 存储单元解码后，把张量字节拷进自己的池段，由分配器给出偏移，并在目录里登记。
3. 非张量字段（uid、tags 等）照旧按 Python 对象存放。
4. 覆盖写（verl v1 在 old 阶段写 `log_probs` 和 `entropy`，ref 阶段又写一次 `log_probs`，driver 会覆盖 `entropy`，见 `trainer_base.py:1506-1516`）：
   - 分配一块新区域，然后改目录；
   - 旧区域不马上释放，等这一步的分区清理（`kv_clear`）时统一回收。这样正在读旧区域的 worker 不会读到被释放的内存。

和现状相比，写入多了第 2 步那一次拷贝（帧 → 池段）。这一跳和 rollout 重叠，不在关键路径上【推断】。

### 3.4 读取路径（NPU worker 快路径）

```
worker ── ZMQ: GET(fields, global_indexes, 要地址) ──> 存储单元
存储单元 ── ZMQ: 张量字段回地址列表；非张量和很小的字段照旧回数据 ──> worker
worker:
  1. 按地址列表算出每个字段的总长度，在本卡 HBM 上分配一块连续缓冲
  2. 每行在缓冲里的位置按 jagged 布局（各行首尾相接）排好
  3. copy_data_batch（DRAM 段用 GH2L）：一批 (源 GVA → 目标 HBM 偏移)，一次提交
  4. 用偏移表在卡上建 jagged 嵌套张量，拼批这一步不再拷贝
  5. 得到 NPU 上的 TensorDict，交给计算
```

- 每次读只要一轮 ZMQ 往返：请求 → 地址。上游 nixl-ucx 实现要两轮（prepare、commit）【已核实】。
- 读取一跳剩下的是：一轮小的控制往返，加一次 DMA【推断】。
- 两台机器在同一超节点内就用 `device_sdma`（走 HCCS，即华为的卡间和超节点高速互联）；否则用 `device_rdma`（NPU 自带的 RoCE 网口，要求全卡 RoCE 互联，A3 最多 8 节点）【已核实，文档】。

### 3.5 读写路径（CPU 进程慢路径）

store 和 driver 读数据时，存储单元从自己的池段取出字节，照旧通过 ZMQ 发回去。

如果池段是存储单元本机的 DRAM，并且 CPU 能直接按指针访问，那么发送时仍然是零拷贝。否则要先用 `GH2H` 拷回主机内存，它内部走 `CopyGH2LH`，同样要借临时 HBM 中转【待验证，见 §6】。所以池段**优先用 DRAM 段**。

### 3.6 哪些字段走快路径

- **走快路径**：张量字段，比如 `input_ids`、`position_ids`、`responses`、`response_mask`、`rollout_log_probs`，以及像素。
- **`multi_modal_inputs`**：现在是每行一个 dict。两种处理办法：
  - 在建行时拆成 `pixel_values`、`image_grid_thw` 两个张量字段（改 `tq_trajectory.py:318`），由 tqbridge 在卡上重新组回 dict；
  - 或者让快路径认得"值是张量的 dict"。
  
  建议用第一种，它对 TQ 是透明的【推断】。
- **留在慢路径**：字符串、标量，以及整批都很小的字段。可以设一个阈值，比如整个字段小于 1 MiB 就照旧回数据，免得为一点数据走一轮 DMA【推断，阈值需实测】。

### 3.7 配置（设想）

```yaml
backend:
  storage_backend: SimpleStorage
  SimpleStorage:
    num_data_storage_units: 2
    payload_transfer:
      backend: memfabric
      protocol: device_sdma        # 不在同一超节点时用 device_rdma
      segment_type: dram           # 存储单元向池贡献的内存类型
      segment_bytes_per_unit: ...  # 按每步数据量和 HCCS 能访问的主机内存定
      fast_path_min_bytes: 1048576
      fallback_to_zmq: true        # MemFabric 出错时退回 ZMQ；上游 nixl-ucx 不回退
```

---

## 4. 与原先 TQ 方案的不同之处

| 方面 | 原先的 TQ 方案 | 方案 2 |
|---|---|---|
| 数据存在哪 | 存储单元进程的普通内存，每行一个张量视图，引用收到的 ZMQ 帧 | 存储单元贡献给 MemFabric 池的一块预先注册的内存段（优先 DRAM），连续存放 |
| 存储单元的目录 | `field → {global_index → 张量对象}` | `field → {global_index → (GVA 偏移, 字节数, dtype, 形状)}`，非张量字段仍存对象 |
| 读取时存储单元回什么 | 数据本身（多帧） | 张量字段回地址列表，非张量和小字段回数据 |
| 读取一跳的传输 | ZMQ over TCP，经主机网卡（同机走回环） | MemFabric `GH2L`（DRAM 段），SDMA 走 HCCS，或 NPU RoCE |
| 读取一跳的 CPU 拷贝 | 两端各一次内核拷贝 + 一次拼批拷贝 | 0 |
| 拼批 | worker 在 CPU 上拼成 jagged 嵌套张量 | DMA 时按偏移直接落成 jagged 布局，卡上建张量 |
| H2D | 每个阶段都要一次 | 没有 |
| 读取的控制往返 | 1 轮（请求 → 数据） | 1 轮（请求 → 地址），之后发一次 DMA |
| 写入一跳 | store → ZMQ → 存储单元，存视图 | 同左，再多一次"帧 → 池段"的拷贝（不在关键路径上） |
| 覆盖写 | 直接替换字典里的引用 | 分配新区域、改目录；旧区域等分区清理时回收 |
| 内存回收 | Python 引用计数，ZMQ 帧随引用释放 | 自己写分配器；按分区或按步回收 |
| 容量上限 | 存储单元所在机器的主机内存 | 预先注册的段大小；DRAM 段还受 HCCS 能访问的主机内存限制（2 TB 的机器约 680 GB，3 TB 的约 2.3 TB，来自 MemFabric 文档） |
| 存储单元进程 | 纯 CPU 的 Ray actor | 要初始化 MemFabric、绑定一个 NPU die |
| 训练 worker 进程 | 只用 ZMQ 客户端 | 还要初始化 MemFabric，作为池的一个成员 |
| store、driver | ZMQ | 不变 |
| tqbridge 返回值 | CPU 上的 TensorDict | NPU 上的 TensorDict |
| 出错时 | ZMQ 报错 | 可以退回 ZMQ（自己设计；上游 nixl-ucx 不回退） |
| TQ 版本 | 0.1.8 | 要有 `PayloadTransfer` 接口：升级到 `rdma` 分支（基于 v0.1.10，头 6a863de，未合入 main），或者把接口移植回 0.1.8 |
| 硬件和软件前提 | 无 | 同一超节点才能用 SDMA；灵衢 L1 ≥1.5.0、HDK ≥25.5.1、固件 ≥7.8.0.6.201、内核 ≥5.10、BIOS 按内存容量升级 |
| **不变的部分** | — | 控制器、元数据、`KVBatchMeta` 的流转、按 `global_index % 单元数` 路由、store 写入的传输方式、driver 读写小字段、非张量字段、ZMQ 控制连接 |

### 和上游 `rdma` 分支 nixl-ucx 实现的区别

方案 2 用的是同一个接口，但数据怎么走不一样：

| | nixl-ucx（上游）【已核实】 | 方案 2 |
|---|---|---|
| 数据在哪 | 主机 DRAM（`mem_type="DRAM"`） | MemFabric 池 → worker 本卡 HBM |
| 谁发起传输 | 发送方单边 RDMA WRITE，推到对方的暂存区 | 读取方发起 `GH2L`，从池里拉 |
| 控制往返 | 每次 put/get 两轮（prepare、commit） | 每次 get 一轮 |
| 内存注册 | 每次发送逐帧注册、用完注销 | 池段启动时注册一次 |
| 接收端 | 从暂存区切出数据，按代码推断会拷一次 | 直接落在最终位置 |
| 失败时 | 报错，不回退 ZMQ | 可回退 ZMQ |

---

## 5. 改动清单

| 仓库 | 文件 | 改什么 |
|---|---|---|
| TransferQueue | `storage/payload_transfer/memfabric.py`（新） | `PayloadTransfer` 实现：快路径 get、地址协议、出错回退 |
| TransferQueue | `storage/payload_transfer/factory.py` | 注册 `memfabric` |
| TransferQueue | `storage/simple_storage.py` | `StorageUnitData` 改成"池段 + 分配器 + 目录"；`_handle_get` 支持回地址 |
| TransferQueue | `storage/bootstrap/simple_storage_bootstrap.py` | 存储单元初始化 MemFabric、贡献内存段；通过 `bootstrap_info` 把池信息发给各个客户端 |
| TransferQueue | `storage/managers/simple_storage_manager.py` | `get_data` 快路径：分配 HBM、提交 `GH2L` 批量拷贝、在卡上拼 jagged 张量 |
| TransferQueue | `config.yaml` | `payload_transfer.memfabric` 的配置项 |
| verl | `utils/transferqueue_utils.py`（tqbridge） | 取数时传入目标设备；对返回 NPU 张量的情况做兼容 |
| agent-lightning | `store/tq_trajectory.py:318` | 可选：把 `multi_modal_inputs` 拆成张量字段 |
| agent-lightning | `cli/store.py:151-169` | TQ 预初始化的配置里加 `payload_transfer` |

---

## 6. 待验证项（方案能否成立）

| # | 问题 | 不成立时怎么办 |
|---|---|---|
| 1 | 存储单元（CPU 的 Ray actor）能不能绑定一个 NPU die，并和该 die 上的训练进程共存 | 把存储单元放到不跑训练的 die 上（要从训练里让出卡），或者改走 KV 方案（方案 4） |
| 2 | 存储单元能不能直接 memcpy 写自己贡献的 DRAM 段，CPU 能不能按指针读它 | 写入改走 `H2GH`（内部 `CopyLH2GH`，借临时 HBM 中转，不在关键路径上）；慢路径读改走 `GH2H`（内部 `CopyGH2LH`） |
| 3 | `GH2L` 的目标 HBM 是否必须预先注册 | worker 预先注册一块固定的接收区，DMA 落进去之后再在卡内拷成张量（卡内拷贝很快） |
| 4 | 只读不写的进程（训练 worker）是否也必须向池贡献内存（MemCache 有这个要求，BM 层没确认） | 每个 worker 贡献一小块 |
| 5 | 每行一个小区域时，`copy_data_batch` 能不能承受几千个条目的批量 | 写入时就让同一字段的相邻行连续存放，读的时候合并成大块 |
| 6 | node09 和 node10 在不在同一超节点（跨机 vnic ping） | 改用 `device_rdma`，要求全卡 RoCE 互联 |
| 7 | 灵衢 L1、固件、内核、BIOS 是否满足要求 | 升级 |
| 8 | `rdma` 分支（v0.1.10 及以后）和 verl 0.9.0 能不能配合 | 把接口移植回 0.1.8 |
| 9 | verl 的计算代码接到已在 NPU 上的 TensorDict 时是否正常（`.to(device)` 应该是空操作） | 在 tqbridge 里补一层处理 |

---

## 7. 实施顺序

1. **M0 先拿到收益上限**：TQ 分支设 `VERL_LOGGING_LEVEL=INFO` 跑 2–3 步，收集每个阶段读取一跳的 `cost time`。数字太小就不做了。
2. **M1 环境**：确认是否同一超节点、软件版本，以及 HCCS 能访问多少主机内存。
3. **M2 最小 MemFabric 脚本**（不碰 TQ）：一个 CPU 进程绑一个 die，贡献一块 DRAM 段，用 memcpy 写入；另一台机器上的 NPU 进程用 `GH2L` 读进 HBM。测带宽，并回答 §6 的第 1–5 项。
4. **M3 原型**：只让 `pixel_values` 走快路径，其余照旧。和现状做 A/B，比较读取一跳的耗时。
5. **M4 全部张量字段走快路径**，加上覆盖写、回收和出错回退。
6. **M5 20 步 A/B**：和 10-31 那一组数据（TQ 155.74 s/步）对比，跑测日志按惯例存进 `~/agl_bench_logs`。

---

## 缩写表

| 缩写 | 全称 | 含义 |
|---|---|---|
| TQ | TransferQueue | 训练数据的分布式暂存与传输层，本文的改造对象 |
| ZMQ | ZeroMQ | 基于 TCP/IPC 的消息库，TQ 现在的传输 |
| DMA | Direct Memory Access | 设备自己读写内存，不用 CPU 逐字节拷贝 |
| SDMA | System DMA | NPU 上的 DMA 引擎，MemFabric `device_sdma` 用它走 HCCS |
| HCCS | Huawei Cache Coherence System | 昇腾卡间和超节点内的高速互联 |
| RoCE | RDMA over Converged Ethernet | 跑在以太网上的 RDMA |
| RDMA | Remote Direct Memory Access | 网卡直接读写远端内存，不经两端的 CPU 和内核 |
| GVA | Global Virtual Address | MemFabric 池里统一编址的全局虚拟地址 |
| G2L / GH2L | Global → Local HBM / Global Host → Local HBM | MemFabric 拷贝类型（`smem_bm_def.h`）：池的 HBM 段 / DRAM 段 → 本机 HBM，一次 DMA |
| H2GH / GH2H | Host → Global Host / Global Host → Host | 本机主机内存 → 池的 DRAM 段 / 反方向，内部借临时 HBM 中转 |
| HBM | High Bandwidth Memory | NPU 上的显存，A3 每 die 64 GB |
| H2D | Host to Device | 主机内存拷到 NPU |
| KV | Key-Value | 键值存储；方案 3–5 的后端类型，本方案不用 |
| BM | Big Memory | MemFabric 的内存池化接口（GVA 分配、拷贝） |
