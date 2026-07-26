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
| 4 | 私有 GWP engine 数学 | 统一 facade、weights/runtime、language/observation/backbone/action、cache/kernels | engine 分阶段 tensor parity |
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

从当前单一 `wam_phase1_skeleton` 过渡到正式但仍不含真实模型计算的构建：

- `wam_core`：公共类型、model/session 生命周期、registry 和 architecture unsupported 入口；PolicySpec loader 与无状态 policy functions 保留到 Slice 2 及后续切片实现；
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

### 6.3 完成状态

Slice 1 已于 2026-07-25 完成：

- 正式 target 为 `wam_core`，并提供 `wam::core` alias；
- `WAM_BUILD_GWP05`、`WAM_BUILD_FASTWAM`、`WAM_BUILD_TESTS` 可独立配置；
- GWP05/FastWAM 只注册 factory，调用时明确返回 `Unsupported`，没有加载 artifact、分配模型资源或生成 action；
- opaque Model/Session 支持 model handle 先释放、已有 session 继续使用、最后一个 session 触发底层 model 延迟销毁；
- public header、基础类型、严格 architecture mapping、registry 和 lifecycle 共 4 个 CTest；
- `gwp_zyj` 中默认模型入口配置和两个模型入口全关闭的 core-only 配置均通过 GCC 11.4 `-Werror` 构建与 4/4 CTest。

## 7. Slice 2：GGUF reader 和 PolicySpec loader

### 7.1 迁入范围

- pinned GGUF/ggml dependency 所需的最小 CMake；
- `src/models/common/gguf_reader.*` 的真实实现；
- `src/policy/policy_spec.*` 中集中实现 draft loader/validator，不增加额外 loader 层；
- metadata fixture writer 和 schema tests；
- converter/inspector 留到真实 architecture contract 和 schema freeze 阶段；Slice 2 使用结构化 GGUF fixture writer 验证同一 reader/loader contract。

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

### 7.4 完成状态

Slice 2 已于 2026-07-25 完成：

- pinned llama.cpp `b9866` 和 SHA256 校验支持在线 archive、本地 archive 或已有 source tree；只构建 GGUF metadata 基础库，不启用 CPU/CUDA model backend 或 release engine patch；
- `GgufReader` 通过 PIMPL 隐藏 ggml/gguf 类型，支持严格 metadata 类型、数组、tensor descriptor/shape、F32 payload 和截断检查；
- `PolicySpecDraft` 覆盖 identity、命名视图、transform/canvas、state、language、normalization stats、action 和 action recovery；
- loader/validator 对 missing/wrong-type key、unknown schema/enum、重复 role、canvas 越界/重叠、统计量 shape、dimension 和 recovery mapping 在模型资源分配前失败；
- `model_load()` 已接通 GGUF open、architecture detection、draft parsing 和 registry dispatch；architecture execution 仍保持明确 `Unsupported`；
- fixture builder 使用 pinned GGUF API 在测试期生成临时 artifact，不提交二进制 GGUF；默认和 core-only 配置均在 `gwp_zyj` 通过 GCC 11.4 `-Werror` 构建及 8/8 CTest。

当前只有合成 draft fixture。六个目标 profile、converter/inspector 一致性和真实 checkpoint cross-validation 必须在对应 architecture Slice 补齐，不能据此声明任何模型/环境组合已支持。

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

### 8.3 完成状态

Slice 3 已于 2026-07-25 完成：

- `artifact.*` 读取 MoT/T5/VAE/scheduler 私有 geometry；PolicySpec 是通用 view/state/action contract 的权威来源，artifact 中仍存在的旧重复字段必须逐项交叉一致；
- 无 `wam.artifact_schema_version` 的 artifact 只允许进入隔离的已审计 GWP05 32D dual-arm legacy migration，固定校验三视图、14/32 state/action、48 horizon、旧 quantile tensor 和 mixed action recovery；legacy 文件没有足够 provenance 可称为 RoboTwin checkpoint，因此 profile 使用中性名称且不对未知 legacy profile 猜测默认值；
- `semantics.*` 和 common scheduler 固定 GWP sequence geometry、prompt canonicalization、MoT token layout、RoPE positions、attention mask 和 flow-match Euler contract，不包含网络计算；
- common input validation 和 `inputs.*` 严格校验 decoded named RGB、flat little-endian F32 state、token/embedding mode、mask、finite payload、可选 `[horizon, model_action_dim]` action noise，并拒绝 batch、history 和编码图像 shortcut；
- GWP factory 返回 metadata-only model，解析 `automatic` backend/precision/language mode 并暴露输入 capability；`Capabilities.action=false`，`session_create()` 明确 `Unsupported`，不分配 weights/GPU、不生成 action；
- 结构化 fixture 同时覆盖 draft/legacy 正例，以及缺失 geometry、私有/通用维度冲突、未知 conversion policy、非法输入 shape/dtype/byte-order/finite/mode；`gwp_zyj` 中 default 与 GWP-only 为 12/12 CTest，core-only 为 9/9 CTest。

Slice 3 的默认 CI 仍只使用小型合成 GGUF。Slice 4A 另行增加可选 external test，并已对四个 22–34 GiB 真实 GWP05 GGUF（legacy F32、MoT BF16、MoT+VAE BF16、packed QKV）验证 metadata-only load、legacy PolicySpec、私有 geometry、conversion-policy tensor count/dtype/name、capability 和 session unsupported 边界；这不等于 engine prediction 已实现。

## 9. 重构前数值 reference

donor 始终保持只读 reference。Slice 4 前固定 prompt、三路图像、raw state、显式 action noise 和 session seed，并记录：

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

### 9.1 Slice 4A Gate 结果

Slice 4A 已冻结一套路径无关的 donor manifest：

- primary oracle 使用真实 legacy F32 GGUF、固定 replay 和显式 `[1,48,32]` action noise，在 donor CPU complete-MoT 路径重新执行；记录 composed image、state、VAE、T5、scheduler、十步 denoise、normalized `[48,32]`、trimmed `[48,14]`、mixed `[48,14]` 和 recovered `[48,14]`；
- prefix/cache 使用既有 audited F32 CUDA capture，因本次容器没有可用 CUDA 而与 primary oracle 明确分开，禁止声称两者来自同一次 execution；
- payload、GGUF、绝对路径和模型全文件 hash 均不提交；`tests/reference/gwp05_donor_reference.json` 只保存输入/中间 tensor 摘要、shape、dtype、byte order、恢复映射和容差；
- `WAM_TEST_GWP05_REAL_GGUF` 和四个 reference path CMake cache 变量使真实 Gate 可选且可重复，默认 CI 不依赖内部资产；本次四种真实 conversion policy 均通过同一 artifact test。

Gate 同时确认两份不能混用的 contract：legacy donor 是 32D + quantile + 上下各半 canvas；当前 `checkpoint_epoch_9_step_100000/transformer_ema` RoboTwin checkpoint 是 14D + z-score，Python reference 使用高位相机占上方 2/3、双腕相机占下方 1/3。后者已由带 `wam.artifact_schema_version=2` 的正式 GGUF 显式描述，不能进入 legacy migration；packed-QKV conversion 后包含 1,424 GWP、242 T5、82 VAE 和 8 stats tensors，并已通过 inspector、PolicySpec/artifact cross-validation 与 metadata lifecycle。该结论只关闭 artifact contract，不替代 BF16/CUDA 数值 Gate。

## 10. Slice 4：完整迁入私有 GWP engine 数学

### 10.1 目标结构

GWP engine 是强耦合私有子图，数值上作为一个 Slice 迁入，但 target 按所有 architecture 共用的职责词汇组织：

```text
src/models/gwp05/
  engine/
    engine.h
    engine.cpp
    engine_internal.h
    weights.cpp
    runtime.cpp
    language_encoder.cpp
    observation_encoder.cpp
    backbone.cpp
    action.cpp
    cache.cpp
  kernels/
    cuda_preprocess.*
```

所有模型的私有 facade 统一为 model-level `create_engine(artifact, options)`、轻量 `create_engine_session(engine, session_options)`、`predict(engine_session, prepared_inputs)` 和 `reset(engine_session)`。统一的是资源所有权和调用角色，不是具体 C++ 类型、跨模型虚基类或通用中间 tensor；公开推理多态仍只有 `ModelImpl`/`SessionImpl`。

### 10.2 Donor 到 target 的职责映射

| donor 文件/内容 | target 职责 |
| --- | --- |
| `text_encoder.cpp` | `language_encoder.cpp`，保留 T5 类型、函数和 stage 名 |
| `vae.cpp` | VAE 数学进入 `observation_encoder.cpp`；临时 image adapter 在 Slice 6 移出 |
| `mot.cpp` joint attention/FFN/RoPE | `backbone.cpp`，保留 MoT 具体实现名 |
| `mot.cpp` action condition/head/denoise | `action.cpp` |
| `weights.cpp` tensor binding/residency | `weights.cpp` |
| `weights.cpp` backend/graph/buffer helpers | `runtime.cpp` |
| `weights.cpp` scheduler/action helpers | `action.cpp` 或已冻结的 common scheduler |
| `cache.cpp` | 可选 `cache.cpp`，只管理 prefix/KV/graph cache 生命周期 |
| `engine.cpp` | `engine.cpp`，只保留配置解析、create/predict/reset 和阶段编排 |

迁移时允许机械移动和私有声明整理。T5、VAE、MoT 等模型组件名称继续存在；不能为了统一文件名重命名 GGUF tensor、改变 graph、合并算子或调整执行顺序。

### 10.3 迁入规则和顺序

- 只迁入神经网络数学、权重绑定、graph、cache 和 kernel；
- 不迁入 donor 的 public API、server、Proto、apps 或 environment adapter；
- 用私有 `engine_internal.h` 明确共享类型；
- 所有 `.cpp` 作为正常编译单元，不恢复 `.cpp` include；
- 不修改公式、执行顺序、tensor shape、scheduler 和精度实现；
- 暂时保留 engine 内原有 preprocessing/postprocessing，以便先验证完整数学；这些临时边界只存在于 GWP 私有目录，并在 Slice 6 删除。
- 先恢复 backend/runtime 与完整 weights binding，再依次对齐 language、observation、backbone 和 action；legacy 32D F32 CPU complete-MoT 全阶段通过后，才恢复 CUDA prefix-cache、BF16 和 packed-QKV 路径。
- `cache.cpp` 是可选 runtime 优化职责，不要求 FastWAM/OpenVLA 等模型创建空文件；一个职责只有在出现可独立验证且拥有独立生命周期的多个 graph 时才继续私有拆分。

### 10.4 Timing 和测试边界

- vision 对应 VAE/vision encoder，text 对应 T5/language path，prefill 对应 prefix/condition/cache build，decode 对应完整 action generation；`model_milliseconds` 是唯一 engine 总计，额外 T5/VAE/MoT/denoise-step 名称进入 `model_timings`。
- cache 命中、external embedding 或 architecture 不存在某阶段时，对应通用 timing 可以为零，不能复用旧请求数值。
- artifact test 保持 metadata-only；weights/runtime structural test 才初始化 backend；component parity 分别检查 language/observation/backbone/action/cache；engine parity 只到 normalized/model-space action。
- test-only stage observer 不进入公共 API，不改变正常 graph；默认 CI 不依赖内部大模型，真实 GGUF/replay 继续通过 CMake 外部路径启用。
- public trim、unnormalize、recovery、`Prediction`、Session lifecycle 和 simulator action 不属于 Slice 4 测试，分别留在 Slice 5/6/7/8。

退出条件：给定相同 prepared inputs，target engine 与 donor 的逐层 tensor 在冻结门槛内一致。Slice 4 不要求公共 `Model::predict()` 已可用。

### 10.5 Slice 4B 完成状态

Slice 4B 已于 2026-07-25 完成以下实现边界：

- donor 的 GWP05 私有数学已迁入正常编译单元：`weights/runtime/language_encoder/observation_encoder/backbone/action/cache`，不再使用 `.cpp` include；`engine_internal.h` 只在该 architecture 内共享 graph、cache、weight 和中间 tensor 类型；
- 私有 facade 在 Slice 4B 已实现 model-level `create_engine()`、`predict()`、`reset()`，Slice 5 增加 architecture-private 轻量 engine session 以拆分共享权重和请求状态；opaque 类型使用私有 deleter 管理，未增加跨模型 engine/encoder/backbone/action 虚基类；
- execution dispatch 支持 legacy/source F32 reference，以及 `mot-bf16-v1`、`mot-vae-bf16-v1`、`mot-vae-bf16-qkv-v1` 的 CUDA latency 配置；unsupported precision/backend/policy 在装载大权重前失败；
- component residency 支持 tokens 全常驻、fixed prompt 先执行 T5 后卸载、external embedding 只常驻 MoT+VAE；reset 清理 graph/prefix/prompt cache、统计和 RNG，并由 session seed 重新初始化；
- engine 输出固定停在 normalized/model-space `[action_chunk, model_action_dim]`，没有恢复 donor 的 trim、action unnormalize 或 raw-state recovery；这些公共 action boundary 仍属于 Slice 5/6；
- `Stats` 映射到 vision/text/prefill/decode/model，并把 VAE、T5、prefix graph、prompt projection、action graph 和逐 denoise step 计时放入 `model_timings`；
- 默认小测试增加 kernel-dispatch contract；真实 external gate 使用 legacy F32 GGUF、三路 PPM、state、外部 T5 embedding 和显式 noise，对 image/VAE/MoT/scheduler/十步 denoise 逐阶段比较，并验证最终 normalized `[48,32]` action、reset 重复性和输入 noise 不被修改；
- `gwp_zyj` 中 CPU `-Werror` build 和 13 个默认 CTest 通过；真实 artifact、冻结 donor manifest 和 Slice 4B engine parity 通过。`WAM_CUDA=ON,WAM_CUDNN=ON,CMAKE_CUDA_ARCHITECTURES=89` 的完整 CUDA/native-BF16 build 与 13 个无设备测试通过。

Slice 4B 结束当时 CUDA driver/NVML 不可用，因此该 Slice 只完成 CUDA/BF16 编译门，没有提前声明 CUDA prefix-cache 或 BF16 数值 parity。后续 Gate A 已在恢复的 A800 上完成 external-embedding CUDA cache/BF16 runtime parity；token/fixed residency 数学虽已迁入并编译，仍继续使用各自独立 fixture。Slice 4B 结束时公开 model 仍为 metadata-only；该边界已在 Slice 5 对 compute backend 打开。

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

### 11.4 完成状态

Slice 5 已于 2026-07-25 完成：

- `Backend::cpu_metadata` 保持 metadata-only，`automatic/cuda` 在 `model_load()` 时创建 model-level engine 并加载一次 backend/weights；
- 私有 engine 增加轻量 `EngineSession`，每个公共 session 独立拥有 graph、prefix/prompt cache、telemetry 和 workspace，共享 immutable weight tensor/backend，并由 model-level mutex 串行提交 backend execution；请求边界的 seed/RNG 在 Slice 6 迁到对应 `Gwp05SessionImpl`；
- 没有恢复 donor 的 `active_session`，创建第二个 session 不重复分配权重；`Capabilities.concurrent_sessions` 继续为 `false`；
- `Gwp05SessionImpl` 接通 `prepare_inputs -> private engine -> Prediction`，显式 noise 不推进 RNG，缺省 noise 使用 `SessionOptions.random_seed`，reset 清 cache 并恢复 seed；
- GWP 私有 legacy adapter 暂时执行 `trim -> quantile unnormalize -> add_current_state`，公共 action 为 F32 `[horizon, real_action_dim]`；该 adapter 在 Slice 6 迁入通用 `action_ops` 后删除；
- 真实 legacy F32 GGUF Gate 同时覆盖 normalized `[48,32]` engine parity、最终 `[48,14]` donor action、两个 session RNG 隔离、reset 复现、model handle 提前释放和单份权重 residency，耗时约 545 秒；默认 13 个无外部资产 CTest 继续通过。

当前 legacy 32D quantile + external embedding F32 CPU reference 与正式 14D z-score RoboTwin + external embedding CUDA/BF16 reference 均完成公共数值 Gate；token/fixed prompt 继续受各自独立 fixture 约束。

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
3. `action_ops` 执行 `trim -> optional clip -> unnormalize -> recover`；
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

### 12.6 实施状态

Slice 6 已于 2026-07-25 完成代码迁移：

- `src/policy/image_ops.*` 实现 canonical named RGB 校验/排序、`none/stretch/cover_center_crop`、nearest/bilinear/bicubic CPU reference、antialias、canvas、pixel range 和 CHW/HWC；schema-v2 antialias 越界 support 使用 PIL-compatible truncate/renormalize，只有隔离的 legacy migration spec 保留历史 clamp；legacy GWP transform 从错误的 `stretch` 修正为 donor 实际使用的 `cover_center_crop`。
- `src/policy/state_ops.*` 生成 normalized/padded model state，同时由具体 session 保留 raw state；`src/policy/action_ops.*` 负责 core action 校验、`trim -> optional clip -> unnormalize -> recover`、最终 Tensor 和 action noise 准备。
- `Gwp05SessionImpl` 持有 `SessionOptions.random_seed` 与 `std::mt19937`；连续缺省请求推进 RNG，显式 noise 不推进，reset 同时清 engine cache 并恢复初始 seed。只有完全默认的 `TensorView` 表示 noise 缺省，半初始化 tensor 必须失败。
- `PreparedInputs` 固定携带 composite image、raw/model state 和完整 action noise；engine 不再读取 raw image、state/action stats 或 raw state，也不再生成随机 noise 或执行最终 action recovery。
- GWP observation encoder 只保留 2x2 patchify、VAE 和 latent normalization。旧 CUDA raw-image resize/canvas/patchify fusion 跨越公共边界，已从 build 删除；以后只能恢复与 CPU reference 对齐的通用 image 实现或 architecture-private patchify 优化。
- `normalization.clip` 已冻结为 min-max/quantile normalized space 的 `[-1,1]`；现有 schema 没有 z-score clip bounds，因此 loader 拒绝 `z_score + clip=true`。当前 14D GWP RoboTwin Python reference 使用 z-score 且不 clip。

当前验证矩阵：core-only 10 项测试通过；CPU GWP 14 项非外部测试通过；CUDA+cuDNN 构建/链接及 14 项非外部测试通过；真实 legacy GGUF artifact、frozen donor manifest 与 legacy F32 complete-engine/public-lifecycle gate 通过；正式 14D z-score RoboTwin packed-BF16 PolicySpec GGUF、Python inspector、C++ artifact/metadata gate 和公开 CUDA/BF16 Session parity 全部通过。A800 上 processed image、normalized state、normalized action、recovered action 的 MAE/max 分别为 `0.00114409/0.00764728`、`0/0`、`0.00140892/0.00781274`、`0.000479634/0.00390983`，显式 noise reset/repeat bit exact，严格 `0.001/0.01` recovered-action PyTorch target 通过。

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

Gate A 已通过，可以进入远程 serving；第二架构仍按 FastWAM Gate B 独立验证。

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

Slice 7 已于 2026-07-25 完成退出条件。正式 GWP05 token-path WebSocket
predict 相对独立 PyTorch recovered action 的 MAE/max 为
`0.000446671/0.00391006`；`robotwin_zyj` 到 `gwp_zyj` 跨容器运行
`beat_block_hammer`，seed `100000` 在第 106 environment step 成功（`1/1`）。
该结果是 integration smoke test，不替代固定 100 次 benchmark success rate。随后使用
SHA256 为 `5e104b97e6a920bc2cc6fbc28aee278f50a5c381e611d46ed22af1968b6b7c57`
的跨模型一致 manifest 完成正式 100 次评测：`87/100`（`87.0%`），共 378 个
action-chunk 请求，无 RPC/runtime 失败。A800 BF16 稳态（每个 episode 第三个及以后
请求）client infer/RPC/server total 的 mean 分别为 `132.29/130.77/124.98 ms`，
p95 分别为 `136.02/134.56/127.46 ms`；峰值 device memory 为 `21.90 GiB`。

## 15. Slice 8：FastWAM 和 eval

### 15.1 FastWAM 首个 profile

优先选择有完整 Python reference、真实 checkpoint 且满足 action-only/action-noise 契约的一个 profile：

1. 冻结 image/state/language/action/noise fixture；
2. 实现 converter、artifact loader、私有 engine 和 SessionImpl；
3. 复用同一 PolicySpec 和 policy ops；
4. 逐层对齐中间 tensor 和最终 action；
5. video/world noise/output 和 joint/video public mode 保持 unsupported。

FastWAM 用于验证公共抽象没有被 GWP 特例塑形。首个 Gate B profile `fastwam_libero_2cam224_minmax` 已完成 converter、1741-tensor GGUF、artifact/semantics/input contract、私有 CUDA/BF16 engine 和公开 Model/Session 纵向接入。固定 `[32,7]` replay 相对 PyTorch 的 MAE 为 `0.000901506`、max 为 `0.00549316`，并与 donor C++ 最终 action 逐 bit 相同。Gate B 同时修正并冻结了 `lower/upper` stats 命名、state output clamp，以及 architecture `norm_eps` 与 policy normalization epsilon 的独立所有权。

FastWAM external-embedding serving 已于 2026-07-26 闭环：RPC wire 继续传原始
instruction，server 根据 PolicySpec 持有 Wan UMT5 tokenizer/encoder 和有界 prompt
cache；C ABI v3 将 BF16 `[128,4096]` embedding 与 I32 mask 传入公共
`EmbeddingInput`。跨 `liberox` client 与 `gwp_zyj` server 的
`libero_spatial/task 0/episode 0` 在 85 个控制步成功，共 9 次 action-chunk 请求；
首请求 `model_text=405.42 ms`，后续 prompt-cache 命中时为 0，全部请求 server total
mean/median 为 `749.42/649.62 ms`。这是单 episode integration gate，不替代后续固定
manifest 的多任务成功率评测。

### 15.2 Eval

实现顺序：

1. RoboTwin server EnvironmentContract 和共享 server core；
2. RoboTwin client observation/action adaptation；
3. `beat_block_hammer` 单 episode；
4. replan/reset；
5. 固定 manifest 100 次（已完成：`87/100`，延迟及失败 seed 已冻结）；
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

Slice 6、Gate A、Slice 7、GWP05 RoboTwin 100 次正式评测，以及 FastWAM LIBERO Gate B 数值、serving 和固定 manifest runner 均已完成。runner 冻结 task/init-state/simulator seed/explicit action-noise seed 和执行参数，逐 episode/request 落盘并支持带 manifest hash、模型身份和 orphan request 检查的恢复运行；task 0 两个 init-state smoke 为 `2/2`。下一步先冻结正式 LIBERO manifest 并扩大到 task 0 的 20 次评测，再扩展到完整 suite；之后进入 FastWAM RoboTwin profile 和 LIBERO-X。
