# wam.cpp 0.5 重构实施计划

## 1. 文档定位

本文档描述如何在当前 `wam.cpp-0.5` 目录中完成 0.5 版本的实际重构。`ARCHITECTURE.md` 是目标架构和契约的权威来源，本文档只负责回答以下问题：

- 从哪一份现有代码开始迁移；
- 按什么顺序把声明骨架变成可运行工程；
- 如何重构 GWP-0.5 engine，同时保证数值行为不变；
- 每个阶段需要提交什么代码、运行什么测试、满足什么退出条件；
- 何时接入 FastWAM、远程 eval 和其余环境。

当本文档与 `ARCHITECTURE.md` 冲突时，以 `ARCHITECTURE.md` 为准，并同步修正文档。尚未冻结的 GGUF key、Proto field number 和 normalization 细节不得在实现中提前视为稳定接口。

## 2. 当前起点

### 2.1 目标目录

`github/wam.cpp-0.5/` 当前是 Phase 1 声明骨架，包含：

- 公共 `wam` 类型和 model/session 接口；
- registry、PolicySpec、image/state/action policy 函数声明；
- GWP-0.5 和 FastWAM 的目录及接口骨架；
- `eval/common/` 与 `eval/sim/` 的入口骨架；
- `ARCHITECTURE.md`。

该目录目前不是完整运行时：没有真实 GGUF 加载、推理 engine、CUDA kernel、服务协议实现和测试体系，CMake 目标也明确标记为 `WAM_PHASE1_DECLARATIONS_ONLY=1`。

### 2.2 代码来源

重构过程中使用三个来源，职责必须区分清楚：

| 来源 | 用途 | 限制 |
| --- | --- | --- |
| `github/wam.cpp-release/` | Git 历史、完整工程基线和发布目录结构 | 可能落后于当前开发代码，不能直接假定它是数值基线 |
| `wam.cpp/` | 已验证的 GWP-0.5 engine、CUDA 路径、工具和测试的主要 donor | 只迁移需要的提交或文件，不能整目录覆盖目标骨架 |
| `github/wam.cpp-0.5/` | 0.5 的目标接口、架构文档和后续唯一开发目录 | 当前只有声明，不能用占位实现代替真实行为 |

### 2.3 总体策略

采用“完整工程基线 -> 冻结现有行为 -> 小步移动边界 -> 第二模型验证 -> 冻结协议 -> 环境集成”的顺序。

不得直接把 `wam.cpp/` 全量复制到目标目录后再集中修正。这样会同时改变构建、API、engine、协议和测试，无法判断数值偏差来自哪一层，也容易覆盖已经讨论确定的 0.5 骨架。

## 3. Git 和工程落地

### 3.1 建立可追踪的 0.5 开发仓库

实施代码修改前先完成以下操作：

1. 保留当前 `github/wam.cpp-0.5/` 的完整可恢复备份。
2. 基于 `github/wam.cpp-release/` 的 `main` 建立 `refactor/0.5` 分支或 worktree。
3. 将 worktree 放到最终目标路径 `github/wam.cpp-0.5/`。
4. 单独迁入当前 `ARCHITECTURE.md`、`plan.md` 和 Phase 1 接口骨架。
5. 把“文档与接口骨架迁入”作为独立提交，不在同一个提交中引入 engine 实现。

如果 release 基线与 `wam.cpp/` 存在差异，应以文件或提交为单位审查后迁入。每次迁入都必须能说明该文件属于构建基础、运行时、GWP engine、工具还是测试，不能产生来源不明的大型合并提交。

### 3.2 建议提交粒度

重构提交按照以下粒度组织：

1. 工程基线和第三方依赖；
2. 当前 GWP-0.5 可运行实现；
3. 当前测试、converter、inspector 和 replay 工具；
4. 0.5 公共类型与 registry；
5. 一个 policy 函数或一个 engine 边界迁移；
6. 对应的 fixture 和测试；
7. FastWAM 单个纵向 profile；
8. 协议和单个环境集成。

每个提交只做一种可验证的结构变化。禁止在移动代码的同时修改 kernel 数学、scheduler 参数、张量布局或默认精度。

## 4. 第一里程碑：恢复完整可运行工程

这一里程碑只解决“目标目录能够构建并运行现有 GWP-0.5”，不改变现有推理数学和输出语义。

### 4.1 迁入内容

- 完整顶层 CMake、安装规则和依赖发现；
- 公共 C++ API、C ABI 和 Proto 支持；
- backend、GGML/GGUF、CUDA 等现有依赖；
- GGUF reader、输入验证、调度器和 replay 基础设施；
- `src/models/gwp05/` 下真实 artifact、semantics、inputs、model、engine 和 CUDA kernel；
- GWP converter、artifact inspector、replay 工具；
- 现有 contract、unit、integration、parity 和 kernel 测试。

FastWAM 在本阶段只保留 factory/registry 的显式 `Unsupported`，不添加伪 engine 或零 action。

### 4.2 基线验证

在 `gwp_zyj` 容器中完成：

- CMake configure 和完整编译；
- CPU/unit/contract 测试；
- CUDA kernel smoke test；
- 当前 GWP-0.5 GGUF 的加载和一次固定输入推理；
- replay fixture 的结果、显存和延迟记录。

退出条件：目标目录可以独立构建和运行；不再依赖 `wam.cpp/` 源码目录参与编译；固定 replay 输出与迁移前一致。

## 5. 第二里程碑：冻结 GWP-0.5 数值基线

在调整 engine 边界前，先建立能够定位误差阶段的 fixture。固定 prompt、三路命名图像、raw state、显式 `action_noise` 和 session seed，并保存或校验以下结果：

1. 按 PolicySpec 顺序取得的图像；
2. resize/crop/composition 后的输入图像；
3. normalized/padded model state；
4. VAE latent；
5. text embedding；
6. VLM/MoT prefix 或 cache 关键输出；
7. 每一个 denoise step 的 action tensor，至少可在诊断模式下比较；
8. engine 输出的 normalized `[48, 32]` action；
9. trim 后的 `[48, 14]` action；
10. 反归一化后的 `[48, 14]` mixed action；
11. 使用 raw state 恢复后的 `[48, 14]` joint-position action。

F32 是重构的 reference 路径，应优先做到 bitwise 一致；如果某个算子无法 bitwise 一致，必须先冻结逐层容差和理由。BF16/CUDA 使用单独容差，不能用最终 action 的宽松容差掩盖中间阶段偏差。

退出条件：任意后续结构改动都能明确判断偏差发生在 preprocessing、模型计算还是 postprocessing。

## 6. 第三里程碑：对齐 0.5 公共接口

### 6.1 公共类型

将完整工程中的旧接口逐项迁移到已经讨论确定的 0.5 形式：

- `Precision` 改为 `ComputePrecision`；
- 使用 `LanguageRuntimeMode::{automatic,tokens,external_embedding}`；
- `ModelOptions` 保存最终 language runtime 选择和可选 fixed prompt；
- `Inputs.noise` 统一命名为 `Inputs.action_noise`；
- `Capabilities.explicit_action_noise` 表示显式 action noise 能力；
- `Stats` 使用 preprocess/model/postprocess/total 和 vision/text/prefill/decode 分解；
- `ModelInfo` 删除 `text_encoder_resident`，只记录最终选择的 `language_mode`；
- `Prediction.action` 始终表示最终完整 `PolicyActionChunk`。

接口迁移时先提供显式编译错误或短期适配层，再逐个修改调用方；禁止让同一个语义长期保留两个字段。

### 6.2 PolicySpec 的公共与私有边界

建议形成以下边界：

```text
include/wam/policy_spec.h
    公共只读 PolicySpec、ImageSpec、StateSpec、ActionSpec、LanguageSpec

src/policy/policy_spec_loader.*
    GGUF metadata 读取、legacy migration、schema 校验、派生字段构造

src/policy/image_ops.*
src/policy/state_ops.*
src/policy/action_ops.*
    只接受已经加载并校验过的 spec，不读取 GGUF，不判断环境或模型名称
```

`GgufReader` 不得出现在公共头文件，也不得在每次 `predict()` 的热路径中重复解析 metadata。Model 加载时只构造一次不可变 PolicySpec，Session 只读取它。

### 6.3 Registry 和生命周期

- `Model::load()` 根据 artifact architecture 选择 factory；
- `ModelImpl` 持有 checkpoint 级共享资源；
- `SessionImpl` 持有请求序列相关的 RNG、cache 和临时执行状态；
- `SessionImpl::predict()` 保持为唯一 architecture 推理多态入口；
- 暂不增加 `PolicyProcessor`、`ModelCoreSession` 或统一 runner 基类；
- 未实现 architecture/profile 必须在大块显存分配前返回 `Unsupported`。

退出条件：公共 API contract、registry dispatch、unsupported path、session reset 和 capability 测试全部通过，现有 GWP 数值尚未发生变化。

## 7. 第四里程碑：重构 GWP-0.5 engine

### 7.1 目标数据流

```text
wam::Inputs
  -> Gwp05 Session 校验语言输入并保留 raw state
  -> PolicySpec 驱动的 image/state preprocessing
  -> Gwp05EngineInputs（全部是模型空间的确定形状 tensor）
  -> Gwp05 engine（权重、VAE、text、MoT、denoise）
  -> normalized/model-space action [horizon, model_action_dim]
  -> PolicySpec 驱动的 trim + unnormalize + recovery
  -> Prediction.action [horizon, real_action_dim]
```

GWP engine 不读取 simulator 名称，不解释 raw observation，不执行 controller 转换，也不返回最终环境 action。GWP session 负责串联通用 policy 函数与私有 engine，但不复制 image/state/action 数学。

### 7.2 先整理编译单元

当前 engine 通过 `#include "weights.cpp"`、`text_encoder.cpp`、`vae.cpp`、`mot.cpp` 等方式拼接实现，类型和资源所有权隐含在 `.cpp` 中。先完成不改变行为的结构整理：

- 增加私有 `engine_internal.h`，只暴露 GWP engine 内部所需类型；
- `weights.cpp`、`text_encoder.cpp`、`vae.cpp`、`mot.cpp`、`cache.cpp` 作为正常编译单元加入 CMake；
- 消除所有 `.cpp` include；
- 不把私有类型移入 `include/wam/`；
- 在这一提交中不修改 tensor shape、执行顺序和数值公式。

完成后先运行全部基线测试，确认这只是构建结构变化。

### 7.3 提取 action postprocessing

第一项行为边界迁移选择 action，因为输入输出和验证最清晰：

1. engine 停止执行 trim、反归一化和 raw-state offset；
2. engine 固定返回 normalized `[horizon, model_action_dim]`；
3. `src/policy/action_ops.*` 按固定顺序执行：
   `trim real dimension -> unnormalize -> recover representation`；
4. GWP RoboTwin 使用 `add_current_state` 和
   `[0,1,2,3,4,5,-1,7,8,9,10,11,12,-1]`；
5. 所有 horizon 行使用同一次请求的 raw state；
6. gripper 两维不添加 state；
7. server 和 simulator client 不得再次恢复 action。

该阶段分别比较 normalized、unnormalized 和 recovered action，不能只比较最终 action。

### 7.4 提取 state preprocessing

- simulator client 只按 `StateSpec.fields` 生成有序 raw state；
- GWP session 保留 raw state，同时调用 `state_ops` 得到 normalized/padded model state；
- engine 只接收 model state，不再读取 normalization statistics；
- statistics 由 GGUF 在 model load 时装入 PolicySpec；
- 缺失 stats、shape 不匹配、非有限值和未知 normalization kind 必须明确失败。

迁移时保留 GWP 当前 quantile 公式及 epsilon，不能顺手替换成 FastWAM 的 z-score/clip 语义。

### 7.5 提取 action noise 所有权

- `Inputs.action_noise` 非空时，校验 little-endian F32 `[horizon, model_action_dim]` 和有限值；
- 未提供时，由 session 的 RNG 按 `SessionOptions.random_seed` 生成；
- 显式 noise 不推进 session RNG；
- `reset()` 清空 cache 并把 RNG 恢复到初始 seed；
- engine 总是接收完整 action noise，不保留内部随机 fallback；
- `Capabilities.explicit_action_noise == false` 时拒绝显式输入。

必须增加连续两次随机预测、显式 noise 插入、reset 后复现和多 session 独立性的测试。

### 7.6 提取 image preprocessing

- client 发送 canonical named RGB images；
- session 按 ImageSpec role 查找并校验视图；
- `image_ops` 执行 per-view resize/crop、canvas composition、pixel range 和 layout；
- engine/VAE 只接收准备好的 composite tensor；
- 图像 key 映射、翻转和 BGR-to-RGB 留在环境 client；
- VAE、patchify 和模型私有 latent 运算仍留在 GWP architecture。

CPU reference 先冻结像素语义，CUDA 优化只在逐像素或冻结容差对齐后启用。相机输入顺序不能影响结果，缺失、重复和未知 role 必须失败。

### 7.7 拆分 model resource 与 session state

目标所有权：

| 所有者 | 内容 |
| --- | --- |
| `Gwp05ModelImpl` | immutable PolicySpec、GGUF context、backend、weights、checkpoint geometry |
| `Gwp05SessionImpl` | seed/RNG、KV/prefix/cache、临时 graph 或 session workspace、输入序列状态 |
| `Gwp05Engine` | 私有模型计算入口，不持有环境状态 |

这一步可能影响显存复用和 cache 生命周期，应晚于 state/action/noise 边界迁移。完成真正的 session 隔离前，`Capabilities.concurrent_sessions` 必须保持 `false`；不能用全局 `active_session` 切换伪装并发支持。

### 7.8 接入 Stats

GWP-0.5 中统计含义固定为：

- `preprocess_milliseconds`：公共 image/state/language 输入准备；
- `model_vision_milliseconds`：VAE/vision encoder；
- `model_text_milliseconds`：text encoder；
- `model_prefill_milliseconds`：VLM/MoT prefix；
- `model_decode_milliseconds`：action denoise；
- `model_milliseconds`：完整模型阶段；
- `postprocess_milliseconds`：trim、反归一化和 action recovery；
- `total_milliseconds`：完整 `predict()`；
- `model_timings`：额外 architecture 私有细分项。

细分项不得被重复累加成第二套总耗时。没有对应阶段的模型可以保留该字段为零或不采集。

### 7.9 GWP 重构退出条件

- engine 输入只包含模型空间 tensor；
- engine 输出固定为 normalized/model-space action；
- GGUF/PolicySpec 只在 model load 时解析；
- session 是 RNG/cache 的明确所有者；
- 最终 action 只在 `SessionImpl::predict()` 内恢复一次；
- CPU F32、BF16/CUDA、replay 和当前 RoboTwin 单任务链路无行为回归；
- 公共层没有 `if gwp05` 或 `if robotwin`；
- GWP 私有 engine 没有环境名称和 controller 逻辑。

完成以上条件后执行 Architecture Review Gate A。

## 8. 第五里程碑：FastWAM 首个纵向 profile

Gate A 通过后再接入 FastWAM。优先选择已有完整 Python reference、真实 checkpoint 且满足 action-only/action-noise 契约的一个环境 profile。

实施顺序：

1. 冻结该 checkpoint 的 image/state/language/action/noise fixture；
2. 实现开发版 converter 和 artifact loader；
3. 实现 FastWAM 私有 engine 和单个 `SessionImpl`；
4. 复用已验证的 PolicySpec、image/state/action 公共函数；
5. 只将 GWP 和 FastWAM 真实共享的代码放入 `src/models/common/`；
6. 逐层对齐 preprocessing、模型中间 tensor、normalized action 和最终 PolicyActionChunk；
7. video/world noise、video/world 输出和 joint/video public mode 保持 `Unsupported`。

FastWAM 首个 profile 的作用不只是增加模型，还要验证公共抽象没有被 GWP 特例塑形。完成后执行 Architecture Review Gate B，再冻结公共 GGUF schema、Proto 字段和主要 ABI。

## 9. 第六里程碑：schema、工具和 RPC

Gate B 后完成：

- 冻结 PolicySpec GGUF key、enum、必填字段和约束；
- converter 只写新 schema，loader 对 legacy GWP 字段显式迁移或交叉校验；
- inspector 打印规范化 PolicySpec，并可按环境 contract 静态检查；
- `ModelInfo`、Proto 和 C ABI 对 PolicySpec/action shape 的序列化保持一致；
- wire 固定为 WebSocket binary frame + Protobuf payload；
- 第一条消息必须为 Hello，并校验 protocol major 和 environment id；
- 一条连接拥有一个 session，同一 session 串行 predict；
- request id 单调递增；Reset、Close 和断线按连接清理 session；
- server 始终返回完整 `[horizon, real_action_dim]` action chunk。

0.5 暂不实现 external-tokenizer manager。依赖未闭环外部 tokenizer 的 profile 在 server 启动阶段返回 `Unsupported`，不能请求期自动下载或猜测路径。

## 10. 第七里程碑：远程 eval 与 RoboTwin

先只打通 RoboTwin 的一个任务，验证协议和资源生命周期，再扩大评测。

### 10.1 Server

`run_robotwin_server.py` 只负责：

- 声明 `environment_id="robotwin"`；
- 提供 RoboTwin EnvironmentContract；
- 解析该环境入口的部署参数；
- 调用 `eval/common/server.py`。

共享 server core 负责模型加载、PolicySpec/EnvironmentContract 权威兼容性检查、连接级 session、Proto 编解码和完整 chunk 返回。server 不解析 RoboTwin 原始 observation key，不硬编码 `[48,14]`，不进行 action recovery。

### 10.2 Client

`run_robotwin_client.py` 负责 RoboTwin 生命周期、raw observation 到 CanonicalObservation 的转换、payload validation、ActionChunkExecutor 和 controller action 转换。模型切换不能改变 client 的模型分支，只替换 server 的 GGUF。

### 10.3 验证顺序

1. codec 和握手单元测试；
2. mock server/client smoke test；
3. `beat_block_hammer` 单 episode；
4. 多次 replan/reset；
5. 固定 manifest 小规模回归；
6. 固定 manifest 100 次评测；
7. 分别报告 runtime parity、integration pass rate、benchmark success rate 和延迟。

checkpoint 未充分微调导致的低成功率不等价于运行时错误，但 action/中间 tensor parity 和协议错误必须先排除。

## 11. 第八里程碑：补齐 FastWAM 和剩余环境

建议顺序：

1. FastWAM RoboTwin；
2. FastWAM LIBERO 或 LIBERO-X 中首个已冻结 fixture 的组合；
3. FastWAM 剩余环境；
4. GWP-0.5 LIBERO/LIBERO-X 权重完成后确认真实视觉 geometry；
5. GWP-0.5 LIBERO；
6. GWP-0.5 LIBERO-X。

每增加一个组合只允许新增或调整：

- converter metadata/statistics；
- 对应 PolicySpec profile；
- 真实需要的 architecture 私有步骤；
- 环境 client observation/action adaptation；
- EnvironmentContract 和 fixture。

不得修改已经冻结的总体调用链，也不得在 session 中增加 `if environment`，或在 client 中增加 `if model`。

## 12. 持续验证矩阵

每个阶段至少运行与改动层次对应的测试：

| 改动 | 必须验证 |
| --- | --- |
| 公共头文件/API | public API compile、C API compile、install consumer |
| PolicySpec/loader | schema 正反例、legacy migration、converter/inspector/runtime 一致性 |
| image ops | Python fixture 像素对齐、role 乱序、缺失/重复 role |
| state ops | normalization/padding fixture、stats shape 和非有限值错误 |
| action ops | trim/unnormalize/recovery 分层 fixture、raw state mapping |
| action noise | 固定 noise parity、RNG 推进、reset 复现、多 session 隔离 |
| GWP engine | F32/BF16/CUDA 中间 tensor 和最终 action parity |
| registry/lifecycle | unsupported、reset、cache、资源释放、并发 capability |
| RPC | Proto round trip、binary-only、Hello-first、request id、断线清理 |
| environment | 单 episode、replan/reset、固定 manifest、成功率与延迟 |

任何阶段失败时先回到最近一个通过的 tensor/协议边界定位问题，不允许通过放宽最终输出容差、补零、默认 identity 或 client/server 重复后处理绕过错误。

## 13. 明确暂不实施

以下内容不属于当前重构主线：

- 通用 `PolicyProcessor`、`ModelCoreSession` 或 `SimulatorRunner` 基类；
- 通用 preprocessing DSL；
- external-tokenizer manager 和自动下载；
- FastWAM video/world noise 与 video/world 输出；
- action ensemble、复杂 replan policy 和通用 gripper filter；
- 在 GWP 边界迁移期间重写 CUDA kernel 或更换 scheduler；
- 在真实 checkpoint fixture 出现前为 pi0/pi0.5/OpenVLA 预建空实现。

这些能力以后只能由真实模型或环境需求驱动，并经过独立架构讨论。

## 14. 下一步可执行任务

下一次开始编码时只执行第一个可验收闭环：

1. 把当前声明骨架安全纳入 Git；
2. 从 release 基线和 `wam.cpp/` donor 迁入完整构建与现有 GWP-0.5 实现；
3. 在 `gwp_zyj` 中恢复编译和现有测试；
4. 建立固定 GWP-0.5 RoboTwin replay baseline；
5. 记录第一里程碑的构建命令、测试结果、GGUF 和 fixture 来源。

在该闭环通过之前，不修改 GWP engine 的 state/image/noise/action 边界。基线通过后，按照第 7 节一次只迁移一个边界，并在每一步提交对应的 parity 测试。
