# wam.cpp 0.6 架构迁移与实施计划

## 1. 总体目标

`wam.cpp` 0.6 采用可破坏式升级：不保留 0.5 API、ABI 和 RPC 协议的兼容分支，但必须保持 GWP05、FastWAM 的数值行为和已经验证的评测能力。冻结当前 0.5 修改后，从基线提交创建 `refactor/0.6` worktree。

0.6 的设计必须始终从使用者和扩展者的角度检查：

- 公共 API 面向使用者，模型内部结构面向实现者，两者不能互相泄漏。
- 用户应能通过最少的配置完成模型检查、加载、推理和服务启动。
- 新增模型不应修改公共 API、Policy、Serving 或 Environment Adapter。
- 新增环境不应修改模型实现、PolicySpec 或 Backend。
- `ModelImpl`/`SessionImpl` 是唯一模型多态边界，不为网络组件建立虚基类。
- Architecture、PolicySpec、Environment Adapter 三个变化维度保持独立。
- 迁移提交一次只处理一种变化：机械移动、接口调整和行为修改不能混在一起。
- 不创建空目录和假想接口，只有真实调用方需要时才增加抽象。
- 目录层次、类型名称和数据流必须能直接表达职责，禁止依赖隐藏约定理解代码。

## 2. 最终目录结构

```text
wam.cpp/
├── include/wam/
│   ├── pipeline.h
│   ├── model.h
│   ├── session.h
│   ├── observation.h
│   ├── prediction.h
│   ├── policy_spec.h
│   ├── runtime_config.h
│   ├── error.h
│   ├── c_api.h
│   └── version.h
│
├── src/
│   ├── artifact/
│   │   ├── gguf_reader.*
│   │   ├── artifact_view.*
│   │   ├── tensor_spec.*
│   │   └── manifest.*
│   │
│   ├── policy/
│   │   ├── policy_spec.*
│   │   ├── observation_processor.*
│   │   ├── image_ops.*
│   │   ├── language_ops.*
│   │   ├── state_ops.*
│   │   ├── action_noise.*
│   │   └── action_decoder.*
│   │
│   ├── runtime/
│   │   ├── model_impl.h
│   │   ├── session_impl.h
│   │   ├── model_registry.*
│   │   ├── runtime_types.h
│   │   ├── logger.*
│   │   └── telemetry.*
│   │
│   ├── backends/ggml/
│   │   ├── backend_context.*
│   │   ├── weight_store.*
│   │   ├── graph_context.*
│   │   ├── tensor_io.*
│   │   ├── graph_ops.*
│   │   └── debug_dump.*
│   │
│   ├── models/
│   │   ├── builtin_modules.cpp
│   │   ├── gwp05/
│   │   │   ├── module.*
│   │   │   ├── contract.*
│   │   │   ├── model.cpp
│   │   │   ├── pipeline.*
│   │   │   ├── resources.h
│   │   │   ├── session_state.h
│   │   │   ├── cache.*
│   │   │   └── networks/
│   │   │       ├── umt5.*
│   │   │       ├── vision_vae.*
│   │   │       └── mot.*
│   │   │
│   │   └── fastwam/
│   │       ├── module.*
│   │       ├── contract.*
│   │       ├── model.cpp
│   │       ├── pipeline.*
│   │       ├── resources.h
│   │       ├── session_state.h
│   │       └── networks/
│   │           ├── vision_vae.*
│   │           ├── proprio_projector.*
│   │           ├── video_dit.*
│   │           └── action_dit.*
│   │
│   └── bindings/
│       ├── c_api.cpp
│       └── metadata_codec.*
│
├── python/wam/
├── proto/
├── apps/
├── serving/
├── adapters/
├── eval/
├── tools/
├── profiles/
├── examples/
├── docs/
├── tests/
├── cmake/
└── patches/
```

目录表示目标职责，不要求提前创建空文件。只有对应实现进入当前迁移阶段时才创建文件。

## 3. 模块职责

### 3.1 `include/wam/`：公开稳定接口

负责：

- 提供 RAII `Model`、`Session` 和高层 `Pipeline`。
- 定义调用方可见的 `Observation`、`Prediction`、`PolicySpec` 和配置类型。
- 定义统一的 `ErrorCode`、`ErrorDetail` 和 `wam::Error`。
- 明确每个 View 类型的所有权、生命周期、dtype、shape 和 layout。
- 提供正式 C ABI v4 头文件和版本信息。

不负责：

- 暴露 GGML、GGUF reader、模型 Geometry、KV cache 或内部 Tensor 名称。
- 暴露 GWP05/FastWAM 私有 prepared input、network output 或 scheduler 类型。
- 通过一个巨大的 `types.h` 收纳所有无关公共概念。

### 3.2 `src/artifact/`：通用模型文件机制

负责：

- `gguf_reader.*` 解析 GGUF header、metadata、Tensor directory 和数据边界。
- `artifact_view.*` 提供类型安全的 metadata/Tensor 查询，不暴露底层解析细节。
- `tensor_spec.*` 表达通用 dtype、shape、required/optional 等 Tensor 约束并执行基础校验。
- `manifest.*` 解析可选 bundle，解析相对 GGUF、tokenizer 和外部 language encoder 资源，并验证 schema、版本和资源路径。
- 在分配大块设备资源前报告结构化文件和 schema 错误。

不负责：

- 判断 GWP05 MoT head 数、FastWAM KV geometry 等模型语义。
- 出现 `if architecture == gwp05/fastwam`。
- 创建 CUDA/GGML backend、上传权重或执行计算图。

不单独创建重复的 `artifact_validator.*`。通用基础校验由 `ArtifactView` 和 `TensorSpec` 完成，模型语义校验由具体 Contract 完成。

### 3.3 `models/*/contract.*`：模型特有 Artifact 契约

负责：

- 声明该 Architecture 必需的 metadata 和 Tensor。
- 从 `ArtifactView` 构造不可变模型 Geometry 和 Tensor contract。
- 校验网络层数、hidden size、head dimension、dtype 和模型组件之间的几何关系。
- 交叉校验模型 Geometry 与公共 PolicySpec 的图像、state、language 和 action 契约。

不负责：

- 解析 GGUF 二进制格式。
- 管理 Backend buffer 或执行网络 forward。
- 读取环境名称、仿真 observation key 或 controller 配置。

依赖方向固定为：

```text
models/*/contract -> src/artifact + src/policy/PolicySpec
```

`src/artifact` 禁止反向依赖具体模型。

### 3.4 `src/policy/`：公共 Policy 语义

负责：

- 加载、验证并保存不可变的公共 PolicySpec。
- 对命名图像执行检查、crop、resize、composition、数值范围和 layout 转换。
- 对 raw state 执行维度校验、padding、mask、normalization，并保留动作恢复需要的 raw state。
- 验证 tokens/embedding 模式、shape 和 attention mask。
- 校验或生成 action noise。
- 按固定顺序执行 action trim、clip、unnormalize 和 representation recovery。
- 返回最终 `[horizon, real_action_dim]` PolicyActionChunk。

不负责：

- T5、VAE、MoT、DiT、scheduler 或 prefix cache。
- 根据模型架构或环境名称分支。
- 解析 GGUF key 或执行环境 controller command。

只有 GWP05 和 FastWAM 都真实使用且语义一致的处理才能进入公共 Policy 层。

### 3.5 `src/runtime/`：生命周期与运行时调度

负责：

- 实现公开 Model/Session/Pipeline 的内部生命周期。
- 定义唯一模型多态边界 `ModelImpl`/`SessionImpl`。
- 根据 `general.architecture` 查询字符串 `ArchitectureDescriptor`。
- 管理 Model 共享资源和 Session 独立状态。
- 保证 Model handle 销毁后，已有 Session 仍可安全持有底层资源。
- 统一日志、Telemetry、Capabilities 和错误转换。

不负责：

- 环境兼容判断、RPC、图像预处理或动作反归一化。
- 通过闭合 `Arch enum + switch` 选择模型。
- 理解网络 Tensor 名称或 GGML graph。

`builtin_modules.cpp` 显式组装内建 descriptor，避免静态初始化顺序和链接器 dead stripping 问题。0.6 暂不实现动态插件。

### 3.6 `src/backends/ggml/`：通用执行基础设施

负责：

- 以 RAII 管理 CPU/CUDA backend、buffer、context、scheduler 和 graph allocator。
- 提供通用 WeightStore、Tensor lookup、Tensor upload/download 和 dtype 转换。
- 提供两个模型已经真实重复使用的基础 graph op。
- 提供统一 debug dump，并由 RuntimeConfig 显式控制。

不负责：

- 解释 PolicySpec、环境或模型 profile。
- 硬编码 GWP05/FastWAM Tensor 名称。
- 包含 UMT5、VAE、MoT、VideoDiT 或 ActionDiT 数学。

只被一个模型使用的 op 保留在该模型目录，不能为了形式统一提前进入 Backend。

### 3.7 `src/models/`：薄模块入口、管线和网络

每个 Architecture 遵循同一分类规则：

- `module.*`：Architecture ID、Capabilities、Contract/Model factory，控制为薄入口。
- `contract.*`：Artifact 与 PolicySpec 的架构规则。
- `model.cpp`：具体 `ModelImpl`/`SessionImpl`，不再建立第二套 Engine/EngineSession 生命周期。
- `pipeline.*`：线性表达一次推理的阶段顺序，不编写底层通用 GGML op。
- `resources.h`：不可变权重、Contract、Backend 和模型共享资源。
- `session_state.h`：RNG、workspace、graph cache 和跨请求可变状态。
- `cache.*`：缓存构建、查找、命中、失效和统计；只在模型真实需要时存在。
- `networks/`：checkpoint 中真实存在的可训练网络 forward。

GWP05：

- `umt5.*` 负责内部语言编码器。
- `vision_vae.*` 负责 observation latent。
- `mot.*` 负责 MoT backbone、state/action embedding 和 action projection。
- Flow scheduler 和 denoise 循环属于 `pipeline.*`。
- Prompt、projected prompt、prefix KV 和 graph cache 属于 `cache.*` 与 `session_state.h`。

FastWAM：

- `vision_vae.*` 负责第一帧视觉 latent。
- `proprio_projector.*` 负责带训练权重的状态投影。
- `video_dit.*` 负责视频上下文 prefill/KV。
- `action_dit.*` 负责每一步 action velocity prediction。
- Flow schedule 和循环属于 `pipeline.*`。

不为 UMT5、VAE、MoT 或 DiT 建立统一虚基类。统一的是目录、资源所有权和数据流规则，不是数学结构。

### 3.8 Binding、Python 与 Serving

`src/bindings/` 负责：

- 实现 C ABI v4 opaque handle。
- 将 C++ Error 稳定映射为 C error code/details。
- 明确输入内存借用和输出内存释放规则。
- 使用单一结构化 metadata codec，禁止手工字符串拼接 JSON。

`python/wam/` 负责：

- 提供正式本地 `Model`、`Session`、`Pipeline` 和远程 `Client`。
- 镜像 C++ 的配置、PolicySpec、Prediction 和异常语义。
- 使用 context manager 管理本地和远程资源。
- Bundle 提供语言资源时允许直接传入原始 instruction。

`serving/` 负责：

- 提供传输无关 service core、WebSocket transport 和 language provider。
- 只调用正式 Python/C API，不访问 C++ internal header。
- 在 server 启动阶段完成 Model PolicySpec 与 EnvironmentContract 的兼容检查。
- 保持一个请求对应一个完整 PolicyActionChunk。

0.6 首个正式版本只支持 Protobuf/WebSocket。gRPC 留到出现真实需求后增加。

### 3.9 Adapter 与 Eval

`adapters/` 负责：

- 将 RoboTwin、LIBERO、LIBERO-X 原始 observation 转换成 CanonicalObservation。
- 将最终 PolicyAction 转换成环境 controller command。
- 描述环境可提供的 camera/state/action contract。
- 只做环境语义转换，不做 checkpoint normalization、padding 或 action recovery。

`eval/` 负责：

- EpisodeSource、Environment lifecycle、ActionChunkExecutor 和 PolicyClient 的组合。
- manifest、随机种子、任务分片、指标、延迟、录像和结果记录。
- 通过正式 API/RPC 调用框架，不 monkeypatch 上游评测函数。
- 把 `execute_steps` 与 checkpoint action horizon 明确区分。

ROS2 不在首轮 0.6 实施范围。出现真实机器人 contract 和测试资产后再创建对应 Adapter。

### 3.10 Apps、Tools、Profiles、Examples 与 Docs

`apps/`：

- `wam-inspect`：无需分配权重，输出 Artifact、PolicySpec、Contract 和资源信息。
- `wam-validate`：完整执行 Artifact/Contract/PolicySpec 校验。
- `wam-predict`：从规范化输入文件完成一次本地预测。
- `wam-serve`：启动正式 WebSocket 服务。

`tools/`：

- 保存 GWP05/FastWAM checkpoint converter 和 conversion report 工具。
- converter 写入唯一权威的 schema v3 PolicySpec，不保留重复 legacy 字段。

`profiles/`：

- 只作为 converter 输入模板和可审查的部署契约来源。
- runtime 不根据 profile 文件名或环境名称推断行为。

`examples/` 和 `docs/`：

- 提供五分钟 Quickstart、本地 C++、本地 Python、远程 Client 示例。
- 提供 Adding a Model、Adding an Environment、Artifact/Bundle、Serving 和 Evaluation 文档。

## 4. 核心依赖规则

```text
apps / serving / adapters / eval / python
                    |
                    v
           Public wam API / C ABI
                    |
                    v
              runtime contracts
                    ^
                    |
          concrete model implementations
             /          |          \
            v           v           v
        artifact       policy     backends/ggml
```

构建层通过独立 target 强制边界：

- `wam_artifact`
- `wam_policy`
- `wam_backend_ggml`
- `wam_runtime_base`
- `wam_model_gwp05`
- `wam_model_fastwam`
- 最终 `wam_core`
- `wam_c_api`

模型 target 可以依赖 Runtime interface、Artifact、Policy 和 Backend；Runtime base 不能依赖具体模型。`wam_core` 通过 `builtin_modules.cpp` 完成最终组装。

禁止规则：

- Core/model 源码中不得出现 `robotwin`、`libero` 或 simulator import。
- Adapter 中不得根据 `gwp05`、`fastwam` 选择预处理。
- Serving 不得执行 normalization、padding 或 action recovery。
- Backend 不得解释 PolicySpec 或模型 Tensor 命名规则。
- Policy 不得 include `GgufReader`、Model Contract 或环境代码。
- Model network 不得读取环境变量、解析 CLI 或直接输出日志。
- 测试以外禁止 include `.cpp` 文件。

## 5. 公共接口决策

### 5.1 C++ API

目标使用方式：

```cpp
wam::Pipeline pipeline = wam::Pipeline::load(model_path, runtime_config);
wam::Prediction prediction = pipeline.predict(observation);
```

高级使用方式：

```cpp
wam::Model model = wam::Model::load(model_path, runtime_config);
wam::Session session = model.create_session(session_config);
wam::Prediction prediction = session.predict(observation);
```

接口规则：

- Model、Session、Pipeline 为 move-only RAII 对象。
- Session 持有共享模型资源，Model 对象先销毁不会使 Session 失效。
- Pipeline 持有一个 Model 和默认 Session，适合单策略常规调用。
- Observation 使用非 owning view 时，调用方必须保证内存在 `predict()` 返回前有效。
- Prediction 拥有输出内存。
- ModelInfo 持有公开只读 PolicySpec 和 Capabilities。
- `reset()` 成功返回 `void`，失败抛出结构化 `wam::Error`。
- 禁止 bool、nullptr、空 Tensor 和 `Status` 混合表达失败。

### 5.2 输出语义

`Session::predict()` 返回的 action 必须已经完成：

```text
model-space output
    -> trim real dimension
    -> optional normalized-space clip
    -> unnormalize
    -> representation recovery
    -> PolicyActionChunk [horizon, real_action_dim]
```

Serving、Python Client 和 Environment Adapter 不得重复上述处理。

### 5.3 RuntimeConfig

RuntimeConfig 显式包含：

- backend：automatic/cpu/cuda。
- compute precision。
- device index。
- CPU thread count。
- cache 容量和开关。
- Logger callback 和 level。
- Debug dump 配置和目录。

SessionConfig 包含：

- random seed。
- Session 私有 cache 开关。

模型实现不得使用隐藏环境变量覆盖这些配置。

### 5.4 Artifact 与 Bundle

- 普通单 GGUF 始终是合法输入。
- 需要 tokenizer 或外部 language encoder 时可使用 bundle 目录。
- Bundle manifest 至少记录 manifest schema、GGUF 相对路径、资源相对路径和资源版本。
- 所有路径必须是 bundle-relative，禁止静默访问 bundle 外部文件。
- 缺少语言资源必须在 Pipeline/Server 创建阶段失败，不能推迟到首次 predict。
- 不允许运行时自动下载 tokenizer 或模型。

## 6. 迁移阶段

### Phase 0：冻结 0.5 基线

状态：**已完成（2026-07-28）**。

- 基线提交固定为 `d20c7ec8e25a8b2c52c5066978cd9a51ec1723e4`，已创建 annotated tag `v0.5-migration-baseline`。
- 已从该 tag 建立 `/testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.6` worktree 和 `refactor/0.6` 分支，0.5 worktree 保持不变。
- 外部资产、checkpoint revision、fixture、模拟器 manifest、参考结果和回归触发规则已记录在 `tests/reference/BASELINE_0_5.md`。
- 基线的 CPU 20 项、CUDA 24 项、Artifact/parity/RPC 和 simulator 结果采用 tagged revision 已完成并记录的 verified checkpoint；Phase 0 不复制大型资产，也不通过重新生成 fixture 改写 oracle。

- 完成当前 0.5 未提交修改的审查和测试。
- 运行 CPU default tests、CUDA build、真实 Artifact gate、GWP05/FastWAM parity 和 RPC tests。
- 提交当前 0.5 修改并标记 `v0.5-migration-baseline`。
- 从该提交创建 `refactor/0.6` worktree。
- 记录外部 GGUF、fixture、checkpoint revision 和参考结果路径，不将大型资产复制进 Git。

验收：基线提交可独立复现现有 GWP05/FastWAM 纵向能力，0.6 的数值回归有明确 oracle。

### Phase 1：建立 0.6 构建和目录骨架

状态：**已完成（2026-07-28）**。

- 项目与公共版本宏已更新为 0.6.0；C ABI 暂时保持 v3，接口重建和 ABI v4 仍属于后续 Phase 2/7。
- 已建立 `wam_artifact`、`wam_policy`、`wam_backend_ggml`、`wam_runtime_base`、`wam_model_gwp05`、`wam_model_fastwam` 和最终 `wam_core` target；`builtin_modules.cpp` 只承担最终内建模型组装。
- 内部实现使用 OBJECT target 表达源码所有权，正式发布仍为单一 `libwam`，未移动或修改 GWP05/FastWAM 数学源码和目录结构。
- 已建立 install/export、`find_package(wam 0.6 CONFIG REQUIRED)` 和独立 external consumer build/run test；CPU 与 CUDA/CUDNN 安装消费均通过。
- 已增加公共头文件独立编译、源码边界、无具体模型 Runtime 配置以及 contract/install 测试；保留现有 unit/integration/rpc，并建立 parity/adapter 测试归档规则。
- 验证结果：默认 CPU 26/26、关闭 GWP05/FastWAM/Serving 的 Runtime-only 15/15、CUDA 12.4 + cuDNN + `sm_80` 26/26；所有构建均为 Release。

- 将项目版本调整为 0.6。
- 建立分层 CMake target 和单向 include/link 规则。
- 建立 install/export、`find_package(wam)` 和最小 external consumer test。
- 建立 contract/unit/integration/parity/rpc/adapter 测试目录。
- 暂不迁移模型数学。

验收：无具体模型时 Runtime fake descriptor、公共头文件和安装消费测试通过。

### Phase 2：重建公共 C++ API

状态：**已完成（2026-07-28）**。

- 已实现 move-only RAII `Model`、`Session` 和 `Pipeline`；`Session` 通过共享模型实现持有资源，公开 `Model` 对象先析构后已有 Session 仍可继续预测。
- 已将公共接口拆分为 `model.h`、`session.h`、`pipeline.h`、`observation.h`、`prediction.h`、`policy_spec.h`、`runtime_config.h` 和 `error.h`，并删除原公共 `types.h` 与实现文件。
- 已将 `PolicySpecDraft` 收敛为公共 `PolicySpec`；`ModelInfo` 通过 `shared_ptr<const PolicySpec>` 暴露模型使用的只读契约，同时公开 `Capabilities`、`Telemetry` 和拥有输出内存的 `PolicyActionChunk`。
- 已统一 C++ 失败语义：`predict()` 返回拥有数据的 `Prediction`，`reset()` 成功返回 `void`，失败抛出 `wam::Error`；公共边界保留已有 `wam::Error`，并将其他异常转换为 `ErrorCode::internal`。
- 已将现有 GWP05、FastWAM 和 C ABI v3 实现迁移到新 C++ 生命周期接口；C ABI v4 与 RPC 破坏式升级仍按计划留在 Phase 7，不在本阶段扩大范围。
- 已增加每个公共头文件的独立编译测试、安装后 external consumer 测试，以及使用者视角的 fake-model 生命周期测试。测试主体只依赖公共 `wam/*` 头文件，覆盖 move-only、moved-from 错误、Model/Session 生命周期、predict/reset、结构化错误保留和非 `wam::Error` 转换。
- 本阶段只调整类型、所有权、调用方式和字段命名；GWP05/FastWAM 数学、计算图、权重加载顺序和目录结构未改变。
- 验证结果：默认 CPU 33/33、关闭 GWP05/FastWAM/Serving 的 Runtime-only 22/22、CUDA 12.4 + cuDNN + `sm_80` 33/33；源码边界和安装消费测试均通过。

- 实现 RAII Model、Session、Pipeline。
- 拆分 Observation、Prediction、PolicySpec、RuntimeConfig 和 Error 公共头文件。
- 将内部 PolicySpecDraft 收敛为公开不可变 PolicySpec。
- 定义 ModelInfo、Capabilities、Telemetry 和 action 输出唯一语义。
- 删除公共 `types.h` 杂物式边界。

验收：最小 C++ 示例无需 internal header 即可加载 fake model、创建 Session、predict/reset，并正确处理生命周期和错误。

### Phase 3：迁移 Artifact、PolicySpec 和 Policy Ops

状态：**已完成（2026-07-28）**。

- 已将 GGUF reader 从 `src/models/common` 迁移到 `src/artifact`，并建立 `ArtifactView`、`TensorSpec` 和严格的 `wam-bundle-v1` manifest。`ArtifactView` 统一接受直接 GGUF 路径或 bundle 目录；manifest 只允许相对路径，拒绝父目录穿越、绝对路径、未知/重复字段和符号链接逃逸，并在打开模型前检查 model、tokenizer 和 external language encoder 资源是否存在。本阶段按既定范围不加入哈希字段或校验逻辑。
- 已明确 Artifact 与模型 Contract 的边界：`src/artifact` 只理解 GGUF、通用 metadata/Tensor 查询、dtype/shape/required 约束和 bundle 资源；GWP05/FastWAM 的网络几何及 Tensor 命名仍由现有模型 Artifact 代码负责，并将在 Phase 5/6 分别改名和收敛为 `contract.*`。Phase 3 不提前移动模型数学或重写 Engine。
- 已将公共 Policy 处理收口为 `observation_processor.*`、`tensor_input.*`、`image_ops.*`、`state_ops.*`、`language_ops.*`、`action_noise.*` 和 `action_decoder.*`。通用 Tensor 输入校验不再位于 `models/common`，action noise 与 action decode 不再混在一个文件中；Policy 和 Artifact 源码边界测试禁止依赖具体模型、环境或反向依赖。
- 已发布 PolicySpec schema v3，将 `wam.input.image.resample_boundary` 变为显式字段。0.6 profile 和 converter 只写 v3；schema v2 仅作为迁移 oracle 只读，缺省为已冻结的 `truncate` 行为，不构成 0.6 legacy Artifact 兼容承诺。
- 已将 PolicySpec 设为图像、state、language 和 action 公共几何的唯一来源。0.6 GWP05/FastWAM converter 不再重复写 image size、camera/view count、state/action dimension、action horizon 和 language context 等 Policy metadata；模型 hidden/layer/head、latent 和 scheduler 参数仍属于模型 Contract。legacy v2 中的重复字段只读取并与 PolicySpec 交叉校验，冲突时立即失败。
- 已将四个正式 profile 升级为 `wam-policy-spec-profile-v2` / Artifact schema v3，并将 GWP05/FastWAM converter revision 升级为对应的 `wam-0.6-*-policy-spec-v3`。测试运行时生成的 0.6 GGUF fixture 已改为 schema v3 且不携带重复 policy 几何，因此无需向仓库提交二进制 fixture。
- 已实现并安装 `wam-inspect` 与 `wam-validate`。`wam-inspect` 只打开 Artifact header、metadata、Tensor directory 和 bundle 资源，输出 architecture、PolicySpec、资源路径和大小信息；`wam-validate` 使用 `cpu_metadata` 执行已编译模型的 Artifact/Policy/Contract 校验，不创建设备 Backend 或上传大块权重。两者成功和失败均输出单行 JSON，错误包含稳定字符串 `code`、数值 `code_value`、message 和字段 details。
- 已增加 Artifact/Policy 单元测试和 CLI 端到端测试，覆盖直接 GGUF、bundle、可选资源、缺失 manifest/resource、非法 JSON/schema、路径穿越、符号链接逃逸、未知字段、损坏 GGUF、Tensor dtype/shape、PolicySpec v2 只读、v3 必填字段和结构化错误。源码边界测试同时保证 Artifact 不依赖 Policy/具体模型、Policy 不依赖具体模型或 GGUF parser。
- 验证结果：默认 CPU 35/35、关闭 GWP05/FastWAM/Serving 的 Runtime-only 24/24、CUDA 12.4 + cuDNN + `sm_80` 35/35；四个 schema v3 profile 均通过 converter profile validator，相关 Python 脚本通过语法编译。外部真实 Artifact/parity gate 未配置资产路径，仍按 Phase 5/6 的数值迁移阶段执行。

验收：合法和损坏 GGUF、错误 metadata/dtype/shape、PolicySpec 冲突、bundle path/resource 缺失均产生稳定结构化错误；inspect/validate 不分配大块模型权重。

### Phase 4：整理 Runtime 与 GGML Backend

- 用字符串 ArchitectureDescriptor 替代 Arch enum/switch。
- 实现显式 builtin descriptor 组装。
- 抽取 BackendContext、WeightStore、GraphContext、Tensor I/O 和公共 debug dump。
- 实现 Logger、Telemetry 和 RuntimeConfig，移除模型隐藏环境变量与 `fprintf`。
- 只抽取 GWP05/FastWAM 已经重复的基础 graph op。

完成状态：

- 已删除 `Arch` enum 及 architecture switch。`ModelRegistry` 现在以严格、区分大小写的字符串 ID 保存 `ArchitectureDescriptor`；descriptor 同时声明 architecture、基础 capabilities 和 factory。`builtin_modules.cpp` 显式组装 GWP05/FastWAM descriptor，dummy `third-model` 测试证明新增架构不需要修改 Runtime enum 或 switch。
- 已增加 GWP05/FastWAM 薄 `module.*` 入口；模型 factory 注册职责不再混入 `model.*`。本阶段只移动注册入口和公共运行时类型，没有迁移模型数学、改变 Tensor 名称、推理次序或 Phase 5/6 目标目录。
- 已建立 `src/backends/ggml/` 公共资源层：`BackendContext` 独占 backend handle，`WeightStore` 独占权重 metadata/buffer，`GraphContext` 独占 graph metadata/allocator，`tensor_io` 统一 F32/BF16 host I/O，`graph_ops` 只包含两个模型确实重复的 dtype cast，`DebugDump` 统一 tensor data/shape metadata 导出。GWP05/FastWAM 已删除相应手工释放和重复实现。
- 已将 `EngineInfo/CoreAction` 收口到 `runtime/runtime_types.h`，新增集中式 `Logger` 和 `Telemetry::append_timing`。GWP05/FastWAM 的错误、进度和运行摘要全部经过 Runtime Logger；模型生产代码不再读取隐藏环境变量，也不再直接写 stdio。
- 已扩充显式 `RuntimeConfig`：debug dump 和 dtype audit 由 `DebugDumpConfig` 控制；prefix/action prompt/prompt KV/graph cache 及 CPU scheduler 调试路径由 `RuntimeTuningConfig` 控制。源码边界测试强制禁止模型层重新引入 `getenv`/stdio，并禁止 Backend 依赖具体模型、Policy 或 Artifact parser。
- 已增加 Runtime/Backend 单元测试，覆盖 Logger level/callback、Telemetry、DebugDump、非法资源参数、Backend move/reset、WeightStore 上传读取、Graph 重复分配异常，以及 32 次完整资源构造/异常/析构循环。既有 Model lifecycle 测试继续覆盖 Session predict/reset/析构路径。
- 验证结果：默认 CPU 36/36、关闭 GWP05/FastWAM/Serving 的 Runtime-only 25/25、CUDA 12.4 + cuDNN + `sm_80` 36/36。外部真实 Artifact/parity gate 未配置资产路径，模型数值迁移仍留在 Phase 5/6。

验收：dummy 第三模型可以只通过 descriptor 接入；Backend 资源在成功、异常、Session reset 和析构路径均无泄漏。

### Phase 5：迁移 GWP05

- 先机械迁移 UMT5、Vision VAE、MoT 数学，提交中不改变数值。
- 将 Artifact 语义重构为 Gwp05Contract。
- 将 Engine/EngineSession 双生命周期压平为 Gwp05 ModelImpl/SessionImpl。
- 拆分 ModelResources、SessionState、Pipeline 和 Cache。
- 将 action encoder/decoder projection 留在 MoT，将 flow loop 放入 Pipeline。
- 将 prompt/prefix/KV/graph cache 放入 Cache 和 SessionState。
- 删除 `engine_internal.h` 总线头文件。

验收：UMT5、VAE、MoT、denoise、cache 分阶段 parity；显式 noise 确定性；reset 恢复 seed/cache；多 Session 生命周期；GWP05 RoboTwin 最终 action parity 和 simulator smoke 通过。

完成状态：

- 已按机械迁移优先原则将原 `engine/` 中的 UMT5、Vision VAE 和 MoT 数学分别移动到 `networks/umt5.*`、`networks/vision_vae.*` 和 `networks/mot.*`；Tensor 名称、算子顺序、flow schedule 和 action projection 数学未做设计性改写。
- 已将 GWP05 的 Artifact 语义收敛为 `Gwp05Contract`。`contract.*` 负责模型特有 metadata、Tensor/geometry 约束和不可变配置；通用 GGUF 读取与 Tensor schema 仍留在 `src/artifact/`。VAE latent 统计值由 Contract 读取，只有创建计算资源时才执行运行时完整性校验，metadata-only inspect/validate 不被计算后端要求污染。
- 已删除旧 `src/models/gwp05/engine/` 和 `engine_internal.h` 总线头文件。`model.cpp` 直接实现 `ModelImpl`/`SessionImpl` 生命周期，不再建立 `Engine/EngineSession` 对象；源码边界测试禁止恢复旧目录、总线头或 GWP05 私有 `engine` namespace。
- 已拆分 `ModelResources`、`SessionState`、`Pipeline`、`Cache`、模型本地 runtime/resource helper 和 network 接口。共享 Backend/WeightStore 由 `ModelResources` 独占，Session 只持有独立可变图与 cache 状态；共享资源上的执行 mutex 明确当前多 Session 可创建但串行执行的线程安全语义。
- `pipeline.*` 只负责资源装配、一次推理的数据流、flow loop 和 telemetry 汇总；prompt/prefix/KV/graph cache 位于 `cache.*`，action encoder/decoder projection 位于 MoT network。`networks/` 不允许依赖 Artifact parser、Policy、Model 或 Serving。
- 已增加显式 noise 不消耗 Session RNG、相同 seed 生成相同 noise、Session cache 隔离、reset 清理 cache/counter 以及旧边界不可恢复的默认测试；既有外部 reference gate 已迁移为 Pipeline/SessionState 接口，并继续覆盖 UMT5、VAE、MoT、denoise、cache、reset 和多 Session 路径。
- 本地验证结果：默认 CPU 36/36、关闭 GWP05/FastWAM/Serving 的 Runtime-only 25/25、CUDA 12.4 + cuDNN + `sm_80` 36/36；安装消费和源码边界测试包含在上述矩阵中。

待完成的外部门禁：

- 当前构建未配置 `WAM_TEST_GWP05_REAL_GGUF`、`WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR`、`WAM_TEST_GWP05_STAGE_ROOT`、RoboTwin GGUF/输入/reference 目录及 donor replay 路径，因此真实权重的分阶段 parity、最终 action parity 和 simulator smoke 尚未执行。
- Phase 5 的结构迁移与默认验收已经完成；在上述真实资产门禁全部通过前，不把 GWP05 数值迁移和 RoboTwin 验收标记为最终完成，也不进入会改变 GWP05 数学的优化。

### Phase 6：迁移 FastWAM

- 机械迁移 Vision VAE、ProprioProjector、VideoDiT、ActionDiT。
- 将 Artifact 语义重构为 FastWamContract。
- 建立 FastWAM ModelResources、SessionState 和线性 Pipeline。
- 将带权重的 proprio projection 从 Pipeline 移入 network。
- 清理重复 `as_f32`、`as_bf16`、`require_weight` 和 Tensor I/O。
- Flow scheduler 留在 Pipeline，只有与 GWP05 语义完全一致时才进入公共层。

验收：FastWAM LIBERO/RoboTwin Artifact gate、frozen replay parity、显式 noise、reset、错误 backend/precision 和 simulator smoke 通过。

完成状态：

- 已按机械迁移优先原则将 Vision VAE、VideoDiT 和 ActionDiT 移入 `networks/`，将 flow scheduler 留在 FastWAM 模型目录；Tensor 名称、图容量、算子顺序、BF16 更新规则和 scheduler 数值未做设计性改写。
- 已将 FastWAM Artifact 语义收敛为 `FastWamContract`。`contract.*` 负责模型特有 metadata、Tensor inventory/shape/dtype、网络 geometry、conversion policy 和不可变 proprio 权重；通用 GGUF/Tensor schema 仍由 `src/artifact/` 负责。
- 已删除旧 `src/models/fastwam/engine/`、`engine_internal.h`、`Engine/EngineSession` 和自定义 deleter。`model.cpp` 直接持有共享 `ModelResources`，每个 `SessionImpl` 独占 `SessionState` 与 RNG；共享资源上的 execution mutex 明确当前多 Session 可创建但串行执行。
- 已建立线性 `pipeline.*`，按 Context/Proprio -> Vision VAE -> VideoDiT prefill -> ActionDiT flow denoise -> action 的顺序表达一次推理。带权重的 proprio projection 已从 Pipeline 移入 `networks/proprio_projector.*`，Pipeline 不再实现网络权重计算。
- `ModelResources` 独占 CUDA BackendContext、WeightStore、Logger 和 DebugDump；`SessionState` 只保存 Session 可变状态。源码边界测试禁止恢复 FastWAM 旧 engine 目录、总线头、私有 `engine` namespace，以及 network 对 Artifact parser、Policy、Model 或 Serving 的反向依赖。
- 三个 network 重复的 `require_weight` 已收敛到模型本地 `networks/ops.*`；F32/BF16 host Tensor 读写统一使用 `backends/ggml/tensor_io.*`，公共 BF16 读写增加了独立 Backend 单元测试。模型专有 I32 position Tensor 保持在 network 内部。
- 已增加纯合成测试，覆盖 proprio projection、错误 backend/precision、SessionState 隔离/reset、显式 noise 不消耗 RNG，以及相同 seed 生成相同 noise；既有 LIBERO frozen replay gate 继续覆盖公开 Action parity、Session RNG 推进、peer Session 和 reset 恢复。
- 本地验证结果：默认 CPU 37/37、关闭 GWP05/FastWAM/Serving 的 Runtime-only 25/25、CUDA 12.4 + cuDNN + `sm_80` 37/37；安装消费和源码边界测试包含在上述矩阵中。

待完成的外部门禁：

- 当前构建未配置 `WAM_TEST_FASTWAM_LIBERO_GGUF`、`WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR` 和 `WAM_TEST_FASTWAM_ROBOTWIN_GGUF`，因此真实 LIBERO/RoboTwin Artifact gate、CUDA frozen replay parity 和 simulator smoke 尚未执行。
- Phase 6 的结构迁移与默认验收已经完成；在真实资产 parity 和至少一个 simulator smoke 通过前，不把 FastWAM 数值迁移标记为最终完成，也不进入会改变 FastWAM 数学的优化。

### Phase 7：C ABI、Python SDK 和用户工具

- 发布 C ABI v4，并明确 struct version、内存 ownership 和错误释放。
- 建立正式 `python/wam` 包和 `pyproject.toml`。
- 提供本地 Model/Session/Pipeline 和远程 Client。
- 支持 GGUF 与 bundle 加载，统一语言资源发现和错误。
- 提供 C++/Python 最小示例与 `wam-predict`。

验收：同一输入/seed 的 C++、C ABI、Python 本地输出一致；wheel 安装、context manager、错误映射和资源释放测试通过。

Phase 7 实施结果：

- C ABI 已破坏式升级为 v4。所有公开输入/输出 struct 均携带 `struct_version` 与 `struct_size`，输入通过对应 `wam_c_*_init` 建立默认值；Model/Session handle、metadata string、Prediction 和 Error 均有唯一匹配的释放函数，所有 free 接受 `NULL`。运行时测试覆盖未知 struct version、结构化错误、metadata GGUF/bundle 加载、字符串/错误/handle ownership 和 metadata-only Session 失败路径。
- C ABI 与 metadata JSON codec 已从 `src/serving` 迁入 `src/bindings`，Binding 只依赖公开 wam API，不再把本地绑定误归类为 Serving transport。安装消费测试同时从纯 C 程序消费安装后的 `wam::c_api`，从 C++ 程序消费 `wam::core`。
- 已建立可构建 wheel 的正式 `python/wam` 包与 `pyproject.toml`。本地 `Model`、`Session`、`Pipeline` 和远程 `Client` 均支持 context manager；原生 ErrorCode、message 与 field-level details 映射为 `WamError`，Python 不再维护 C ABI v3 struct 布局。
- `Pipeline.load` 同时接受 GGUF 与 bundle；`LanguageResources` 以显式参数优先、bundle manifest 次之的顺序发现 tokenizer/language encoder，禁止资源路径逃逸且不自动下载。`Pipeline.predict(..., instruction=...)` 延迟建立正式 language provider，也允许调用方直接传入 prepared tokens 或 BF16 embedding。`eval/common/native.py`、`language.py` 和 `rpc.py` 仅保留指向正式 SDK 的兼容导出，不再复制 native/client/provider 实现。
- 已提供可编译的 C++ 最小示例、Python 原始 instruction 示例、`docs/python-sdk.md` 和 wheel console script `wam-predict`。CLI 从规范 NPZ 读取 `state`、`image.<role>` 与 raw/prepared language input，输出 JSON 或 NPY action chunk。
- 本地验证结果：默认 CPU 41/41、关闭 GWP05/FastWAM/Serving 的 Runtime-only 27/27、CUDA 12.4 + cuDNN + `sm_80` 41/41。wheel 测试在临时源码副本中完成构建、隔离 venv 安装、正式包导入和 `wam-predict --help`，不会向源码树写入 build/egg-info 中间文件。

Phase 7 尚未执行的外部门禁：

- 当前构建仍未配置真实 GWP05/FastWAM GGUF、冻结 observation/noise 与独立 reference action，因此不能把“相同输入/seed 的 C++ direct、C ABI 与 Python local 数值一致”标记为已验证。Python local 直接调用 C ABI v4，结构、错误和生命周期路径已覆盖；真实数值 parity 必须复用 Phase 5/6 的外部 Gate A/Gate B 资产后再完成。
- 远程 `Client` 在 Phase 7 正式化了用户 API 和资源生命周期，但仍封装现有 `wam.rpc.v05` wire protocol。`wam.rpc.v06`、transport-neutral service core 和 C++/Python local/WebSocket remote 的真实数值一致性属于 Phase 8，不能在本阶段提前宣称完成。

### Phase 8：Serving、Adapter 和 Eval

- 将 Proto 升级为 `wam.rpc.v06`。
- 拆出 transport-neutral service core 和 WebSocket transport。
- 将 native bridge、RPC client、language provider 从 `eval/common` 移到正式模块。
- 为 RoboTwin、LIBERO、LIBERO-X 建立独立 Adapter。
- 建立共享 ActionChunkExecutor、manifest、metrics、video 和 result writer。
- 删除 runner 中 architecture 分支、重复 compatibility check 和 monkeypatch。

验收：RPC malformed request、握手、reset、关闭、错误映射和并发能力测试通过；本地与远程 Prediction 一致；三个环境至少完成一个固定种子 smoke episode。

### Phase 9：文档、打包与发布验收

- 重写 README，提供五分钟构建、inspect、local predict 和 serve 流程。
- 增加 Adding a Model、Adding an Environment、Artifact Bundle、Serving 和 Evaluation 文档。
- 增加 CPU CI、安装消费测试和可选 CUDA/external gate 说明。
- 更新 benchmark、支持矩阵和已知限制。
- 清理过期 Slice 文档、旧协议、legacy converter 和重复脚本。

验收：新用户只阅读 README 和示例即可完成本地/远程首次预测；扩展者只阅读对应指南即可新增模型或环境，不需要阅读 GWP05/FastWAM 全部实现。

## 7. 测试体系

### 7.1 默认无外部资产测试

- 公共头文件独立编译。
- 外部 CMake consumer install/find_package。
- Error、RuntimeConfig、Model/Session 生命周期。
- Registry descriptor 和 unsupported Architecture。
- GGUF reader、ArtifactView、TensorSpec、manifest。
- PolicySpec 成功/失败和所有 Policy ops。
- Backend RAII、Tensor I/O、GraphContext 异常清理。
- GWP05/FastWAM 合成 Contract 和 geometry。
- C ABI、Python SDK、RPC、Adapter fixture。

### 7.2 外部 Artifact 与数值 Gate

- GWP05 legacy/reference 资产仅作为迁移 oracle，不作为 0.6 runtime 兼容承诺。
- 0.6 schema v3 GWP05 RoboTwin Artifact 和分阶段 parity。
- 0.6 schema v3 FastWAM LIBERO/RoboTwin Artifact 和 replay parity。
- 所有外部路径必须显式配置、绝对存在并成组校验。
- 大模型、checkpoint、replay、视频和 rollout 结果不进入 Git。

### 7.3 端到端 Gate

- C++ direct、Python local、WebSocket remote 输出一致。
- Model 先释放、Session 继续预测。
- reset 后显式/隐式 noise 行为可复现。
- 不支持的 backend/precision/language mode 在资源分配前失败。
- EnvironmentContract 不兼容时在第一个 episode 前失败。
- ActionChunkExecutor 的 horizon/execute_steps/queue/reset 行为正确。

## 8. 可读性和可维护性门槛

- `module.cpp` 只包含 descriptor 和 factory，不承担推理逻辑。
- `pipeline.cpp` 必须能从上到下读出推理阶段，不隐藏到多层 callback。
- `resources.h` 与 `session_state.h` 必须明确不可变共享资源和可变 Session 状态。
- Network 文件不得处理 GGUF、CLI、Environment、RPC 或 Policy action decode。
- 禁止出现同时定义配置、权重、Backend、graph、cache、Session 和全部 op 的 God header。
- 复杂函数必须使用强类型输入输出，禁止使用无语义的 `std::vector<float>` 在多个阶段间裸传递。
- 所有公开 API 和关键内部类型说明 ownership、shape、dtype 和线程安全语义。
- 注释解释约束和原因，不逐行复述代码。
- 不通过合并大文件追求“文件少”，也不为单一实现机械创建 `.h/.cpp` 对。

## 9. 用户体验验收

以下使用路径必须清晰且有自动测试：

1. `wam-inspect model.gguf`：无需 GPU 查看 Architecture、PolicySpec 和模型要求。
2. `wam-validate model.gguf`：在加载权重前得到明确校验结果。
3. C++ `Pipeline::load -> predict`：最短本地推理路径。
4. Python `Pipeline.load -> predict`：支持 context manager 和原始 instruction。
5. `wam-serve --model ...`：启动通用服务，不要求手工指定 Architecture、action dim 或 camera count。
6. Python remote Client：连接、握手、predict、reset、close。
7. Eval runner：通过 Adapter 和 ActionChunkExecutor 完成可复现 episode。

错误必须包含稳定 ErrorCode、可执行的 message 和字段级 details，不能只返回“inference failed”。

## 10. 0.6 明确不做的内容

- 不实现动态模型插件和运行时共享库加载。
- 不实现通用 DAG ExecutionPlan 或 FeatureStore。
- 不实现 gRPC。
- 不实现 ROS2，直到存在真实机器人纵向测试。
- 不实现多 GPU、offload 或自动设备切分。
- 不实现运行时自动下载 tokenizer/checkpoint。
- 不为未来假想模型建立 VAE/Backbone/ActionHead 虚接口。
- 不保证 0.5 C++ API、C ABI v3、RPC v0.5 或 legacy Artifact 兼容。

## 11. 发布完成定义

0.6 只有同时满足以下条件才能发布：

- 目标目录和依赖规则落实，旧 `src/models/common`、双 Engine 生命周期和 God header 已消失。
- GWP05 与 FastWAM 均通过 Artifact、Contract、数值 parity 和至少一个 simulator smoke。
- C++、C ABI、Python local 和 WebSocket remote 行为一致。
- 默认 CPU 测试、安装消费测试和 Python package 测试全绿。
- 所有公共 RuntimeConfig 能替代模型环境变量和直接日志输出。
- README、Examples、Adding a Model、Adding an Environment 文档完成。
- 新用户能够在五分钟路径内完成模型检查并理解下一步操作。
- 新增模型和环境的修改范围通过依赖测试验证，没有反向污染核心层。
