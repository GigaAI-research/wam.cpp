# wam.cpp 0.5 重构实施计划

## 1. 文档定位

本文档描述如何从干净的 0.5 声明骨架逐步实现可运行的 `wam.cpp`。`ARCHITECTURE.md` 是目标架构和契约的权威来源，`MIGRATION_INVENTORY.md` 记录 release/donor 文件盘点和迁入归属，本文档负责实施顺序、阶段边界和验收门槛。

当三份文档冲突时，先更新设计结论，再修改代码。GGUF key、Proto field number、normalization 公式和公开 ABI 在对应 review gate 前都属于 development draft。

## 2. 当前起点

### 2.1 开发目录和 Git

唯一开发目录是：

```text
/testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5
```

它是 `refactor/0.5` worktree，日常提交推送到私有 `private-origin`；公开仓库是只允许 fetch 的 `public-upstream`。`wam.cpp-0.5-skeleton-backup/` 只作为只读安全副本，不在其中开发。

干净起点由以下内容组成：

- 56 个原始 Phase 1 骨架文件；
- `MIGRATION_INVENTORY.md`；
- public release 的 Apache-2.0 `LICENSE`；
- 扩充后的 `.gitignore`。

旧 apps、serving、Proto、integration、dependency 脚本和未声明 engine 子文件不在起点中。它们仍可从 `public-upstream/main@7adca01` 或 donor 查阅，但不会被隐式编译或整体恢复。

### 2.2 代码来源

| 来源 | 用途 | 约束 |
| --- | --- | --- |
| `github/wam.cpp-release/` | Git provenance、Apache-2.0 release 参考、已有文件历史 | 不整体恢复旧 runtime/API/serving |
| `wam.cpp/` | 当前可运行 GWP-0.5 数值 reference、engine 数学、测试和 replay 工具 | 不全树复制，不上传内部 replay payload |
| `github/wam.cpp-0.5/` | 0.5 接口和唯一实现目录 | 每个新增文件必须属于明确 Slice |

### 2.3 Skeleton-first 原则

实施顺序固定为：

```text
clean skeleton
-> build/test foundation
-> GGUF + PolicySpec
-> GWP artifact contract
-> private GWP engine
-> GWP Model/Session vertical path
-> policy boundary extraction
-> new serving
-> FastWAM and eval
```

不先恢复一套完整旧工程再重构。旧代码只按可编译功能切片迁入，并在迁入时放到 0.5 已定义的所有权边界。不得使用全树 rsync、旧 public API 兼容副本或旧 server 帮助新 runtime 表面跑通。

“逐步迁入”不等于逐个 `.cpp` 文件复制。强耦合的 GWP engine 数学作为一个私有子图整体迁入；公共 framework、artifact contract、policy functions 和 serving 分别独立验收。

## 3. 通用实施规则

1. 每个 Slice 结束时主分支必须可 configure、build 和运行该 Slice 声明的测试。
2. 尚未实现的 model/profile/path 必须返回 `Unsupported` 或 `NotImplemented`，禁止补零或伪 action。
3. 移动 engine 数学时不同时修改公式、tensor layout、scheduler、kernel 或默认精度。
4. checkpoint、GGUF、内部 replay、生成 action、profile 和构建输出不进入 Git。
5. 真实 replay 通过显式外部路径使用，不能硬编码 donor 或用户目录。
6. 公共层不出现 environment/model 交叉分支；模型层不读取 environment id。
7. 一个提交只完成一个可验证结构变化，并同时增加或调整相应测试。
8. FastWAM 未实现前保持显式 unsupported，不用空骨架表示已支持。

## 4. Slice 总览

| Slice | 目标 | 主要产物 | 退出门槛 |
| --- | --- | --- | --- |
| 0 | 干净骨架 | 56 个骨架文件、迁入清单、LICENSE、ignore 规则 | 无隐式旧实现 |
| 1 | 构建系统和测试框架 | 正式 core target、最小 contract tests | CPU configure/build/CTest |
| 2 | GGUF reader + PolicySpec loader | metadata reader、draft loader/validator | schema 正反例通过 |
| 3 | GWP artifact/semantics/input | 不加载大权重的 GWP contract path | fixture load/validate 通过 |
| 4 | 私有 GWP engine 数学 | weights/VAE/text/MoT/cache/kernels | engine 中间 tensor parity |
| 5 | GWP Model/Session 纵向调用 | registry -> model -> session -> predict | 最终 action parity |
| 6 | action/state/noise/image 边界 | PolicySpec 驱动的完整 0.5 GWP path | Gate A 通过 |
| 7 | 新 Proto/WebSocket serving | binary wire、连接级 session | 协议和 lifecycle 测试通过 |
| 8 | FastWAM 和 eval | 第二架构、Gate B、环境远程评测 | profile parity + rollout |

## 5. Slice 0：干净骨架

### 5.1 内容

- 保留 Phase 1 的 include、registry、policy、GWP/FastWAM 和 eval 声明骨架；
- 保留 `ARCHITECTURE.md`、本计划、迁入清单和 Apache-2.0 LICENSE；
- 删除 worktree 从 release 隐式继承的旧实现；
- `.gitignore` 排除构建、模型、GGUF、Python cache、内部 replay 和 upstream simulator checkout。

### 5.2 验证

- target 文件集合与 skeleton backup 一致，额外只允许迁入清单和 LICENSE；
- `git diff --check` 通过；
- Markdown fence 完整；
- Phase 1 declaration-only CMake 不引用已删除文件；
- 删除内容均可从 Git 历史恢复。

## 6. Slice 1：构建系统和测试框架

### 6.1 构建目标

从当前单一 `wam_phase1_skeleton` 过渡到正式但仍可不含真实模型计算的构建：

- `wam_core`：公共 model/session、registry、PolicySpec 和无状态 policy functions；
- contract/unit test target；
- 明确的 `WAM_BUILD_GWP05`、`WAM_BUILD_FASTWAM`、`WAM_BUILD_TESTS` 选项；
- 严格 warning 和 include dependency 检查；
- 安装/Proto/C ABI 暂不进入 Slice 1。

构建文件以 0.5 目标重新组织，可以参考 release 的 CMake dependency 处理，但不能直接恢复旧顶层 CMake，因为它会重新引入旧 apps、serving、Proto 和 GWP source list。

### 6.2 测试

先建立最小测试目录：

```text
tests/
  CMakeLists.txt
  contracts/
  unit/
  support/
```

Slice 1 覆盖：

- public header 自包含；
- registry dispatch；
- 未实现 architecture/profile fail-fast；
- Model/Session 所有权和 reset 声明契约；
- FastWAM 未实现路径不分配大块资源。

退出条件：在不下载 checkpoint、不使用 replay、不需要 GPU 的情况下，`gwp_zyj` 中 CPU configure/build/CTest 通过。

## 7. Slice 2：GGUF reader 和 PolicySpec loader

### 7.1 迁入范围

- pinned GGUF/ggml dependency 所需的最小 CMake；
- `src/models/common/gguf_reader.*` 的真实实现；
- `src/policy/policy_spec_loader.*`；
- metadata fixture writer 和 schema tests；
- converter/inspector 暂只实现验证 Slice 2 所需的最小开发入口。

### 7.2 边界

- `GgufReader` 只存在于内部代码；
- public header 不依赖 ggml、CUDA 或 Protobuf；
- metadata 在 model load 时解析一次，predict 热路径不读取 GGUF；
- loader 产生不可变、已校验的 PolicySpec；
- 旧 GWP metadata 迁移集中在 loader/architecture artifact 层，不进入公共 policy ops。

### 7.3 验证

- 缺失 required key；
- 未知 enum；
- image role 重复；
- canvas placement 越界/重叠；
- state/action dimension 冲突；
- normalization stats shape 错误；
- action recovery mapping 长度和索引错误；
- converter/inspector/loader 对同一 fixture 的解释一致。

Slice 2 使用 development draft key，不冻结公开 artifact schema。

## 8. Slice 3：GWP artifact、semantics 和 input contract

### 8.1 迁入范围

从 donor/release 参考以下职责，但按 0.5 接口实现：

```text
src/models/gwp05/artifact.*
src/models/gwp05/semantics.*
src/models/gwp05/inputs.*
src/models/common/input_validation.*
src/models/common/scheduler.*
```

Slice 3 不迁入 weights、VAE、text、MoT、cache 或 CUDA kernel。

### 8.2 行为

- 根据 GGUF architecture/profile 构造 GWP 私有 geometry；
- 从 legacy metadata 构造 draft PolicySpec 并交叉校验；
- 校验 named images、state、language 和可选 action noise；
- 暴露 capabilities，但 predict 仍明确 unsupported；
- 无大权重/GPU 条件下完成 artifact contract 测试。

退出条件：GWP artifact 可以完成 metadata-only load/validate，所有缺失、shape 和 unsupported 错误在资源分配前发生。

## 9. 重构前数值 reference

donor `/testessfs10/users/yejun.zeng/codes/gwp/wam.cpp` 始终保持只读 reference。Slice 4 前固定 prompt、三路图像、raw state、显式 action noise 和 session seed，并记录：

1. composed image；
2. normalized/padded state；
3. VAE latent；
4. text embedding；
5. prefix/cache 输出；
6. 每个 denoise step action；
7. normalized `[48,32]` action；
8. trim/unnormalize 后 `[48,14]` mixed action；
9. raw-state recovery 后 `[48,14]` joint-position action。

内部 fixture 保留在 donor/local storage，通过命令行或 CMake cache path 提供，不提交到个人私有 GitHub。可以提交不含内部路径和 payload 的 shape、dtype、容差与摘要 manifest。

F32 优先要求 bitwise；无法 bitwise 时先冻结逐层容差和原因。BF16/CUDA 使用独立容差，最终 action 宽松容差不能掩盖中间 tensor 偏差。

## 10. Slice 4：完整迁入私有 GWP engine 数学

### 10.1 迁入范围

GWP engine 是强耦合私有子图，作为一个 Slice 迁入：

```text
src/models/gwp05/engine/engine.*
src/models/gwp05/engine/engine_internal.h
src/models/gwp05/engine/weights.cpp
src/models/gwp05/engine/text_encoder.cpp
src/models/gwp05/engine/vae.cpp
src/models/gwp05/engine/mot.cpp
src/models/gwp05/engine/cache.cpp
src/models/gwp05/kernels/cuda_preprocess.*
```

### 10.2 迁入规则

- 只迁入神经网络数学、权重绑定、graph、cache 和 kernel；
- 不迁入 donor 的 public API、server、Proto、apps 或 environment adapter；
- 用私有 `engine_internal.h` 明确共享类型；
- 所有 `.cpp` 作为正常编译单元，不恢复 `.cpp` include；
- 不修改公式、执行顺序、tensor shape、scheduler 和精度实现；
- 暂时保留 engine 内原有 preprocessing/postprocessing，以便先验证完整数学；这些临时边界只存在于 GWP 私有目录，并在 Slice 6 删除。

退出条件：给定相同 prepared inputs，target engine 与 donor 的逐层 tensor 在冻结门槛内一致。Slice 4 不要求公共 `Model::predict()` 已可用。

## 11. Slice 5：GWP Model/Session 纵向调用

### 11.1 数据流

```text
Model::load
  -> registry selects gwp05
  -> Gwp05ModelImpl owns immutable artifact/backend/weights
  -> session_create owns RNG/cache/request state
  -> Gwp05SessionImpl::predict
  -> private Gwp05Engine
  -> Prediction
```

### 11.2 所有权

| 所有者 | 内容 |
| --- | --- |
| `Gwp05ModelImpl` | immutable PolicySpec、GGUF context、backend、weights、checkpoint geometry |
| `Gwp05SessionImpl` | seed/RNG、prefix/cache、session workspace、请求序列状态 |
| `Gwp05Engine` | 私有模型计算入口，不持有环境状态 |

完成真正 session 隔离前，`Capabilities.concurrent_sessions` 保持 `false`。不得恢复全局 `active_session` 切换并把它声明成并发支持。

### 11.3 验证

- load/create/predict/reset/free lifecycle；
- 两个 session 的 seed/cache 隔离；
- reset 后复现；
- 资源释放和加载失败清理；
- 固定显式 noise 下最终 action 与 donor 对齐；
- runtime 不读取 environment id。

Slice 5 可以通过 GWP 私有 legacy boundary adapter 暂时产生 donor 等价输出，但 adapter 必须有明确删除任务，且不能进入 public API/server/client。

## 12. Slice 6：迁移 policy 边界

目标数据流：

```text
wam::Inputs
  -> named image/state/language validation
  -> PolicySpec image/state preprocessing
  -> Gwp05EngineInputs
  -> private GWP engine
  -> normalized/model-space action
  -> PolicySpec trim + unnormalize + recovery
  -> Prediction.action [horizon, real_action_dim]
```

### 12.1 Action postprocess

1. engine 停止 trim、反归一化和 raw-state offset；
2. engine 返回 `[horizon, model_action_dim]` normalized action；
3. `action_ops` 执行 `trim -> unnormalize -> recover`；
4. GWP RoboTwin mapping 为 `[0,1,2,3,4,5,-1,7,8,9,10,11,12,-1]`；
5. 所有 horizon 行使用本次请求同一份 raw state；
6. server/client 不重复恢复。

分别比较 normalized、trimmed、unnormalized 和 recovered action。

### 12.2 State preprocess

- session 同时保留 raw state 和 normalized/padded model state；
- engine 只接收 model state，不读取 normalization stats；
- stats 在 model load 时进入 PolicySpec；
- 保持 GWP quantile 公式和 epsilon，不替换成 FastWAM z-score；
- 缺失 stats、shape 错误和非有限值明确失败。

### 12.3 Action noise

- 显式输入严格为 little-endian F32 `[horizon, model_action_dim]`；
- 缺省时由 session RNG 根据 `SessionOptions.random_seed` 生成；
- 显式 noise 不推进 RNG；
- reset 恢复初始 seed 并清 cache；
- engine 总是接收完整 noise，不保留随机 fallback；
- capability 为 false 时拒绝显式 noise。

### 12.4 Image preprocess

- client 发送 canonical named RGB images；
- `image_ops` 按 role 完成 resize/crop/canvas/pixel range/layout；
- engine/VAE 只接收 prepared composite；
- 环境 key、flip 和 BGR-to-RGB 留在 client；
- VAE/patchify 留在 architecture；
- CPU reference 像素语义先冻结，CUDA 实现随后对齐。

### 12.5 Stats

- preprocess：公共 image/state/language 准备；
- vision：VAE/vision；
- text：text encoder；
- prefill：VLM/MoT prefix；
- decode：action denoise；
- postprocess：trim/unnormalize/recovery；
- model/total：各自唯一总计；
- `model_timings`：额外 architecture 私有细分。

## 13. Review Gate A

Slice 6 结束后必须确认：

- engine 输入仅为模型空间 tensor，输出仅为 normalized/model-space action；
- PolicySpec 只在 model load 时解析；
- session 是 RNG/cache 所有者；
- 最终 action 只在 `SessionImpl::predict()` 内恢复一次；
- 公共层没有 `if gwp05`/`if robotwin`；
- GWP engine 没有 environment/controller 逻辑；
- F32/BF16/CUDA 与 donor 无行为回归；
- 没有 `PolicyProcessor` 或 `ModelCoreSession` 强制多态层。

Gate A 通过后才能实现远程 serving 或第二架构。

## 14. Slice 7：新 Proto/WebSocket serving

Slice 7 从 0.5 wire contract 新建，不恢复旧 gRPC server/Proto：

- WebSocket binary frame + Protobuf payload；
- Hello-first，校验 protocol major 和 environment id；
- 一条连接拥有一个 session；
- 同一 session 串行 predict，最多一个 in-flight；
- request id 单调递增；
- Reset/Close/断线清理连接 session；
- server 返回完整 `[horizon, real_action_dim]` chunk；
- server 不执行 action recovery/controller conversion。

Slice 7 可以使用 development Proto field number 验证 wire mechanics，但在 FastWAM Gate B 前不承诺稳定 schema。旧 `scripts/serving/`、`proto/wam.proto` 和 GigaWorld-Policy integration 仅作参考，不直接迁入。

退出条件：Proto round trip、binary-only、Hello-first、错误 request id、多连接隔离、reset 和断线清理测试通过。

## 15. Slice 8：FastWAM 和 eval

### 15.1 FastWAM 首个 profile

优先选择有完整 Python reference、真实 checkpoint 且满足 action-only/action-noise 契约的一个 profile：

1. 冻结 image/state/language/action/noise fixture；
2. 实现 converter、artifact loader、私有 engine 和 SessionImpl；
3. 复用同一 PolicySpec 和 policy ops；
4. 逐层对齐中间 tensor 和最终 action；
5. video/world noise/output 和 joint/video public mode 保持 unsupported。

FastWAM 用于验证公共抽象没有被 GWP 特例塑形。Gate B 检查公共函数没有 architecture/environment 分支，并在通过后冻结 PolicySpec GGUF key、Proto field number 和主要 ABI。Gate B 前生成的 development artifact 必须重新转换。

### 15.2 Eval

实现顺序：

1. RoboTwin server EnvironmentContract 和共享 server core；
2. RoboTwin client observation/action adaptation；
3. `beat_block_hammer` 单 episode；
4. replan/reset；
5. 固定 manifest 100 次；
6. FastWAM RoboTwin；
7. FastWAM LIBERO/LIBERO-X；
8. GWP LIBERO/LIBERO-X 权重完成后确认真实 geometry 再适配。

server 做权威 PolicySpec/EnvironmentContract compatibility check；client 持有 simulator lifecycle、CanonicalObservation、ActionChunkExecutor 和 controller conversion。client 不判断 model，session 不判断 environment。

发布结果分别报告 runtime parity、integration pass rate、benchmark success rate、延迟和 checkpoint 来源。低成功率不自动等价于 runtime 错误。

## 16. 持续验证矩阵

| 改动 | 必须验证 |
| --- | --- |
| build/public API | clean configure、header compile、registry/unsupported |
| GGUF/PolicySpec | schema 正反例、converter/inspector/loader 一致性 |
| artifact/input | metadata-only load、shape、dtype、finite 和 capability |
| engine | F32/BF16/CUDA 中间 tensor parity |
| action ops | trim/unnormalize/recovery 分层 fixture |
| state ops | normalization/padding/stats shape |
| action noise | 固定 noise、RNG 推进、reset、多 session |
| image ops | 像素 reference、role 乱序/缺失/重复 |
| lifecycle | load/create/predict/reset/free 和失败清理 |
| RPC | Proto round trip、binary-only、Hello、request id、断线 |
| environment | 单 episode、replan/reset、manifest、成功率/延迟 |

失败时回到最近通过的 tensor/协议边界定位，不通过放宽最终容差、补零、默认 identity 或重复后处理绕过。

## 17. 明确暂不实施

- 通用 `PolicyProcessor`、`ModelCoreSession`、`SimulatorRunner`；
- preprocessing DSL；
- external-tokenizer manager 和自动下载；
- FastWAM video/world public input/output；
- action ensemble、复杂 replan 和通用 gripper filter；
- GWP 迁入时重写 kernel/scheduler；
- 没有真实 fixture 的 pi0/pi0.5/OpenVLA 空实现。

## 18. 下一步

Slice 0 cleanup commit 完成后进入 Slice 1。第一项代码工作是审查当前 declaration-only CMake 和 public headers，设计不依赖旧 runtime 的正式 `wam_core`、最小 test target 和 unsupported registry path；此时不迁入 GGUF、GWP engine、Proto 或 eval 实现。
