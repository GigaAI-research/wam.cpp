# wam.cpp 0.5 框架优化设计

## 1. 文档状态

本文档定义 `wam.cpp` 0.5 版本的目标架构、模块职责、改造范围、实施顺序和验收标准。后续关于 GGUF 字段、动作语义、相机角色、协议版本等细节的讨论，应继续更新本文档；细节冻结后，再以本文档作为代码优化依据。

当前状态：`Draft`。

本文中的源码路径均相对于 `wam.cpp` 仓库根目录。本文描述的是 0.5 目标状态，不代表对应代码已经实现。

## 2. 版本目标

0.5 的核心目标是让两个模型架构通过统一运行时适配三个仿真环境：

| 模型 | LIBERO | LIBERO-X | RoboTwin |
| --- | --- | --- | --- |
| GWP-0.5 (`gwp05`) | 目标支持；权重训练中 | 目标支持；权重训练中 | 目标支持；权重已可用 |
| FastWAM (`fastwam`) | 目标支持；权重已可用 | 目标支持；权重已可用 | 目标支持；权重已可用 |

“支持”分成四个层次：

1. artifact 和接口支持：GGUF 可以完整描述 checkpoint 的输入输出契约，runtime 可以加载并校验。
2. 数值支持：C++ 与对应 PyTorch checkpoint 在冻结输入上完成中间结果或最终 action 对齐。
3. 集成支持：环境可以完成 reset、观测转换、推理、action chunk 执行和 episode 结束。
4. 效果支持：在对应 benchmark 上报告成功率。

前三项是 0.5 对某个模型/环境组合正式声明“支持”的工程发布门槛，并且数值支持和集成支持必须使用该组合的真实 checkpoint 完成。第四项依赖 checkpoint 训练质量，必须报告，但低成功率本身不等价于运行时实现错误。checkpoint 尚未完成时，可以先完成 schema、adapter 和合成契约测试作为阶段性进度，但不能用这些测试替代真实 checkpoint 的数值对齐和 simulator rollout，也不能提前把该组合标记为正式支持。

GWP-0.5 的 LIBERO/LIBERO-X 权重完成后，首先确认其视觉图结构是原生两视图、动态视图还是其他明确结构，并冻结对应的 VAE/composition geometry。converter 不能只根据 `num_views=2` 猜测模型语义。FastWAM 三个环境的现有权重需要在 Phase 0 记录准确 checkpoint revision、模型变体、输入输出契约和 normalization statistics。

0.5 不要求实现 `pi0`、`pi0.5` 或其他 VLA，但本次设计不得把模型核心重新绑定到具体环境，避免后续接入这些模型时再次重构公共框架。

## 3. 关键设计结论

### 3.1 三个维度必须独立

框架需要区分三个概念：

- `architecture`：神经网络结构和计算图，例如 `gwp05`、`fastwam`。
- `policy profile`：某个 checkpoint 的部署契约，例如双视图、8 维 state、7 维增量 EEF action、特定归一化统计。
- `environment adapter`：仿真环境原始观测和控制接口，例如 LIBERO、LIBERO-X、RoboTwin。

它们之间不能通过硬编码分支绑定：

- `src/models/gwp05/` 和 `src/models/fastwam/` 中不能出现 `if libero` 或 `if robotwin`。
- LIBERO、LIBERO-X、RoboTwin adapter 中不能根据 `gwp05` 或 `fastwam` 选择不同 parser。
- server 不能通过手工指定 `--arch`、`--num-views` 或 `--action-dim` 重建 checkpoint 契约。
- converter 可以使用模型和训练配置生成 profile，但运行时只消费已经写入 GGUF 的结果。

三个 environment-specific server 启动入口不违反上述独立性：它们只选择一个数据化 environment contract 并交给共享 compatibility checker，不能选择 architecture 计算路径、修改 PolicySpec 或实现 checkpoint 数学。

### 3.2 GGUF 表示 checkpoint 的部署契约

每个 GGUF 默认只描述一个明确的 policy profile。典型 artifact 名称可以是：

```text
gwp05-libero-panda.gguf
gwp05-liberox-panda.gguf
gwp05-robotwin-dual-arm.gguf
fastwam-libero-panda.gguf
fastwam-liberox-panda.gguf
fastwam-robotwin-dual-arm.gguf
```

文件名只方便人类识别，runtime 不从文件名推断行为。checkpoint 的部署契约、normalization statistics 和模型 geometry 必须来自 GGUF metadata/tensor。0.5 暂不设计外部 tokenizer 的发现、下载、打包、校验、缓存和生命周期；需要这套能力的模型/profile 在部署闭环确定前保持 `Unsupported`，不能由 client 或 server 静默下载一个 tokenizer 猜测运行。

如果 LIBERO 和 LIBERO-X 实际使用完全相同的 checkpoint、相机契约、state/action 语义和归一化统计，它们可以共享一个 GGUF。环境 adapter 仍然分别存在，因为两套仿真环境的原始字段、图像方向、episode 流程和控制接口不相同。

### 3.3 GGUF 环境名称只作为 provenance

可以在 GGUF 中保存：

```text
wam.policy.training_dataset = "libero"
wam.policy.profile = "libero_panda_2cam_eef"
wam.policy.embodiment = "panda"
```

但 `training_dataset` 和 `profile` 不能直接驱动模型分支。实际行为必须由明确字段决定，例如相机角色、图像布局、state/action 维度、归一化方式和动作表示。

serving 握手中的 `environment_id` 是独立的部署字段，用于确认 client 连接了正确的 environment-specific server，并选择该 server 已注册的 environment contract。它不从 GGUF 的 `training_dataset/profile` 推导，也不能驱动 C++ model 分支；兼容性必须通过 environment contract 与 PolicySpec 的结构化字段比较得到。

### 3.4 输出边界必须唯一

runtime 返回的 action 定义为：

> 已完成 checkpoint 所需的裁剪、mask、反归一化和训练动作表示恢复，但尚未转换成具体 simulator controller 调用格式的 `PolicyAction`。

例如，模型输出张量从 padded 32 维裁剪到真实维度、使用 checkpoint 的 q01/q99 反归一化，以及将训练时的 mixed delta/absolute action 恢复为最终 `joint_position`，都属于 policy 后处理；将夹爪符号转换成 LIBERO 的约定属于环境 adapter；实际调用 RoboTwin `take_action(..., action_type="qpos")` 属于 simulator runner/client。

如果 checkpoint 训练动作语义与框架定义的 `PolicyAction` 语义不同，转换规则必须由 policy profile 显式描述。禁止在环境 adapter 中重复 checkpoint normalization，也禁止在模型核心中调用仿真器语义。

真实 action 恢复位于一次完整的 `SessionImpl::predict()` 调用内，但不属于 architecture 私有神经网络 graph。具体 session 在 engine 产生 normalized/model-space action 后调用 PolicySpec 驱动的无状态 action 函数，并在返回 `Prediction` 前完成恢复。恢复函数可以读取本次请求中尚未 normalization/padding 的 raw state；server 和 simulator client 均不得补做这一步。该阶段计入 `Stats.postprocess_milliseconds`，不计入 action denoise 的 `model_decode_milliseconds`。

当前 GWP-0.5 RoboTwin checkpoint 的恢复语义冻结为：先把 `[horizon, model_action_dim]` 裁剪到 `real_action_dim=14`，再使用 checkpoint action statistics 反归一化，最后只对左右臂的 12 个关节维加上同一次 `predict()` 输入的 raw state；两个 gripper 维保持反归一化后的绝对值。chunk 中所有 timestep 都引用同一份请求时刻 raw state，与现有 Python reference 行为一致。

一次 `predict()` 始终返回完整的 `PolicyActionChunk`，公共 shape 为 `[horizon, real_action_dim]`。`horizon` 是 checkpoint 输出契约；每次实际执行多少步由 simulator side 的 ActionChunkExecutor 决定，更复杂的 replan/ensemble 属于后续控制策略，均不能改变 runtime 输出边界。

### 3.5 对 vla.cpp 的借鉴边界

参考 [VinRobotics/vla.cpp](https://github.com/VinRobotics/vla.cpp) 时，0.5 借鉴以下做法：

- 通过 GGUF architecture metadata 自动选择 model factory；
- 使用公共 images、language、state、action noise 输入和动态 action 输出；
- server core 只加载模型并提供统一预测协议；环境差异只由三个 server 启动入口提供的 environment contract 表达，不进入 C++ model/policy 层；
- 将 checkpoint 的真实 state/action dimension 和 normalization statistics 尽量随 artifact 保存。

以下现状不应照搬：

- client 通过 `ARCH_PRESETS` 和大量 `if arch` 决定 tokenizer、image size 和预处理；
- LIBERO parser registry 通过 architecture 选择环境 parser，形成模型和环境的交叉依赖；
- 图像只依赖位置而没有稳定逻辑角色；
- 部分模型仍要求外部 stats JSON 或环境变量选择 embodiment；
- Python client 与各模型 C++ 实现分别承担 resize、state/action normalization 和 gripper 后处理，公共 `predict()` 的 normalized/unnormalized 输出边界不完全一致。

vla.cpp 的 server 返回完整 action chunk，Python client 将其 reshape 后把前 `n_action_steps` 放入队列，每个 `env.step()` 消费一行，队列为空再重新推理；`n_action_steps` 和 checkpoint horizon 是两个不同概念。wam.cpp 0.5 保留“runtime 返回完整 chunk”的做法，但统一 action 后处理边界，并把最小 chunk 执行状态放到 simulator 一侧的 ActionChunkExecutor。

vla.cpp 中 Pi0.5、SmolVLA、BitVLA 等实现已经在各自 C++ `predict()` 尾部完成 action 反归一化，通用 server 只转发返回值；0.5 沿用这一正确边界。其 GWP 实验性 `wam-server` 曾在 serving 层硬编码 `[48,32] -> [48,14]`、normalization statistics、state offset 和 delta mask，这种特殊路径不应照搬：0.5 必须让直接 C++ 调用与远程 server 调用具有相同的最终 `PolicyActionChunk` 语义。

因此，wam.cpp 0.5 在 vla.cpp 的通用 runtime/registry 基础上增加 self-describing PolicySpec、命名相机角色、统一 action 边界和独立环境适配，但不复制其 client 侧的 architecture 分支。新增能力以数据契约和小型无状态函数实现，不再为 preprocessing 与模型计算建立强制的双层多态框架。

### 3.6 最小抽象原则

0.5 对新增抽象设置以下准入规则：

- architecture 的唯一运行时多态入口是现有 `ModelImpl`/`SessionImpl`；一次推理不再拆成必须跨模型实现的 processor/core 两套虚接口。
- PolicySpec 是不可变数据契约，不是可编程 pipeline，也不负责调度模型步骤。
- 只有 GWP-0.5 和 FastWAM 两条真实纵向路径都使用且语义一致的处理，才进入公共 `src/policy/` 函数；单模型逻辑留在对应 architecture 目录。
- Python common 层同样遵循“至少两个真实调用方”原则，不预先创建 generic runner/client 或 adapter 基类。
- 新抽象必须减少已经出现的重复或隔离已证实的变化轴，不能只为假设中的 pi0/pi0.5 或未来 simulator 预留空接口。

该原则保留 vla.cpp 单一 `predict()` 主链路的可读性，同时通过 PolicySpec 和环境 adaptation 修复其模型/环境交叉依赖。

## 4. 目标运行链路

```text
upstream checkpoint + training config + dataset statistics
                         |
                         v
                 model converter
                         |
                         v
             GGUF + conversion report
                         |
                         v
simulator runner/client
       |
       | raw observation
       v
observation adapter -> CanonicalObservation protobuf
       |
       v
WebSocket binary frame
       |
       v
environment-specific server entry
compatibility check + request bridge -> wam::Inputs
       |
       v
ModelImpl / SessionImpl::predict()
  |- PolicySpec-driven common policy_io functions
  |  image/state numerical preprocessing
  |- architecture-specific computation
  |  gwp05 or fastwam graph/cache/denoise
  `- common policy_io action decode
     trim/mask/unnormalization
       |
       v
PolicyActionChunk [horizon, real_action_dim]
       |
       | transport
       v
ActionChunkExecutor (simulator side)
chunk queue/execute_steps
       |
       v
action adapter -> controller command
       |
       v
simulator runner/client env.step()
```

0.5 的三个 simulator 统一采用 client/server 分离。持有 simulator 的 Python client 负责 observation/action adaptation、环境生命周期和 ActionChunkExecutor，并将 `CanonicalObservation` 作为 Protobuf payload 通过 WebSocket binary frame 发送。RoboTwin、LIBERO 和 LIBERO-X 分别使用自己的 server 启动入口；入口向共享 server core 注册固定的 environment id/contract，由 server 在启动和握手阶段完成 checkpoint/environment compatibility check，再构造 `wam::Inputs` 并调用具体 `SessionImpl`。server 始终返回完整 `PolicyActionChunk`。当前 RoboTwin“依次执行返回 chunk 中全部 action”的行为等价于 `execute_steps=horizon`，LIBERO 和 LIBERO-X 使用同一 wire 边界。

environment/profile compatibility 的权威判断只在 server 侧执行，client 不再各自复制一套 PolicySpec 兼容规则。client 在握手中声明 environment id，并继续对每次实际 observation 和返回 action 做本地 shape/字段完整性检查；这类 payload validation 不等于 compatibility 决策。换 checkpoint 时只替换对应环境 server 的 `--model xxx.gguf`，server 必须在第一个 episode 前接受或拒绝，不能通过命令行同步覆盖 action 维度、相机数量或 normalization 参数。真机评测后续可以在 `eval/` 下增加与 `sim/` 平级的 server/client 入口，并复用相同 wire/session 边界，不要求 simulator 和真机 runner 继承统一基类。

## 5. 模块职责

### 5.1 Artifact Conversion and Validation

这是离线构建工具链，不参与每次 `predict()`。它负责把 upstream checkpoint 编译成 runtime 可以直接加载和解释的 GGUF，并在转换期和加载期建立一致的结构与语义校验。

负责：

- 从 upstream checkpoint、训练配置和 dataset statistics 提取权重与部署契约。
- 将模型特有 geometry、通用 `PolicySpec` 和 normalization tensor 写入 GGUF。
- 在写文件前验证 metadata、tensor shape/dtype、normalization statistics 和 profile 之间的一致性。
- 生成可选 conversion report，记录来源路径或 checkpoint revision、converter revision、profile 和 component telemetry。
- 与 runtime 共用同一套 schema 规则，确保 converter 生成的 artifact 不需要额外环境 YAML 或 stats JSON 才能解释。

0.5 必须执行的校验：

- GGUF header、schema version 和 `general.architecture`；
- metadata 必填字段、枚举值和字段间约束；
- tensor offset、byte size 和文件边界；
- tensor shape/dtype 与 architecture geometry；
- PolicySpec 的 view、state、action、language 契约；
- normalization tensor 的存在性、shape 和有限值；
- PolicySpec 与模型 input/output head 的一致性。

0.5 暂不实现：

- 完整 GGUF 文件的 SHA256 或其他全文件 hash；
- source checkpoint、converter 工具和 patch 的全文件 hash；
- 签名 manifest、artifact authenticity 或供应链签名验证；
- 模型每次启动时为了计算 hash 而完整扫描所有 weight payload。

现有 `ModelInfo.artifact_sha256` 和 Proto 字段为兼容保留，0.5 返回空字符串。server、cache、replay 和 adapter compatibility 不能依赖该字段。需要记录 artifact identity 时，0.5 使用 architecture、policy profile、checkpoint revision、artifact schema version 和文件大小；这不是字节级唯一身份，但足以满足当前开发和仿真范围。

不负责：

- 仿真环境原始字典字段映射。
- rollout 的 `execute_steps`、replan/ensemble、最大 episode 步数和成功条件。
- 通过文件名暗示 runtime 行为。

### 5.2 PolicySpec

`PolicySpec` 是 checkpoint 的公共部署契约，由 GGUF 加载。它回答“调用方必须提供什么、runtime 返回什么”，不描述网络内部如何计算，也不描述 simulator 如何执行。0.5 将其分成以下部分：

| 子结构 | 内容 |
| --- | --- |
| `PolicyIdentity` | artifact schema version、profile id、checkpoint revision；training dataset 和 embodiment 只作为 provenance |
| `ImageSpec` | 有序的逻辑视图记录、每视图 resize/crop/interpolation/目标尺寸、可选 composition、颜色空间、pixel range 和 tensor layout |
| `StateSpec` | 字段顺序、real/model dimension、padding、normalization、mask、clip 和 epsilon |
| `LanguageSpec` | token/embedding 输入模式、prompt template、最大长度、padding/truncation 和 attention mask；外部 tokenizer 资源部署暂不纳入 0.5 |
| `ActionSpec` | horizon、real/model dimension、字段顺序、最终 representation/frame、gripper encoding、normalization、裁剪/mask，以及必填的 `ActionRecoverySpec` |

0.5 只把 GWP-0.5/FastWAM 跨环境部署真实需要的外部契约放入 PolicySpec。runtime 是否接受显式 action noise、是否支持外部 embedding、并发 session 等可选调用模式由 `ModelInfo.capabilities` 表达；runtime 自行生成 action noise 所需的私有 geometry、denoise scheduler、history/cache geometry 等保留在 architecture 私有 metadata。调用方可见的 action noise shape 由 ActionSpec 的 horizon 和 model action dimension 推导。将来如果某个 checkpoint 要求调用方必须提供 history 或 temporal input，再以真实 fixture 为依据扩展 schema，不在 0.5 预留通用 preprocessing DSL。

图像预处理契约随 checkpoint 写入 GGUF metadata，实际 observation 图像仍通过每次请求的 `Inputs.images` 传入，不写入 artifact。模型加载时由 artifact loader 一次性执行 `GGUF metadata -> ImageSpec -> validation`，成功后 model/session 持有解析完成的只读 `ImageSpec`。预测热路径中的公共 image 函数只接收 `ImageSpec` 和命名图像，不直接查询 `GgufReader`，也不根据 architecture、environment 或文件名推断预处理行为。

`ActionRecoverySpec` 只描述从已经裁剪、反归一化的 checkpoint action 到最终 `PolicyAction` 所需的最小确定性恢复。0.5 支持 `identity` 和 `add_current_state` 两种 kind；后者通过 `reference_state_indices[real_action_dim]` 指定每个 action 维引用哪个 raw state 维，值 `-1` 表示该维不加 state。该索引数组是 state-relative 恢复的唯一信息源，不再同时保存可由它推导的 `delta_mask`。

同一个公共概念只能有一个权威来源。新 artifact schema 中，action horizon、real/model action dimension、real/model state dimension 和有序 image view 记录由 `PolicySpec` 定义；image count 和 role order 从 view 记录推导，不再同时保存 `count`、`roles` 和 indexed `role` 三份信息。不得再由 `gwp05.action_chunk`、`fastwam.action_dim` 等独立字段重复定义公共契约。迁移期可以读取模型私有旧字段并与 PolicySpec 交叉校验；architecture 内部可以持有从 PolicySpec 派生的 geometry，但不能成为第二个配置源。

`PolicySpec` 不保存仿真器原始 key，例如 `observation/head_camera/rgb`；不保存 `env.step()`、episode 长度、success 判断、`execute_steps`、replan 或 ensemble；也不保存 backend、device、precision 和 transport 配置。这些分别属于环境集成代码、ActionChunkExecutor 或后续控制策略扩展，以及 runtime deployment config。

### 5.3 Environment Adaptation

每个仿真环境提供两个可独立部署的方向，不强制继承统一基类，也不要求它们是同一个有状态对象：

- observation adapter：从环境原始 observation 中提取图像、state 和 instruction，将相机映射为逻辑角色，完成 quaternion 到 axis-angle 等环境语义转换，并按 `StateSpec.fields` 产生尚未 normalization/padding 的 raw state。
- action adapter：把 `PolicyAction` 的 representation、frame 和 gripper encoding 转换成 simulator controller command，但不重复 checkpoint action decode。
- client payload validation：验证本次实际 observation/action payload 的必填字段、shape 和 dtype；不决定某个 checkpoint 是否适配该环境。
- reset hook：只清理当前方向确实拥有的状态；跨时间 action queue 由 ActionChunkExecutor 持有。

0.5 的 RoboTwin、LIBERO 和 LIBERO-X client 都在 simulator 侧组合两个方向。三个环境的 server 启动入口分别声明 environment id 和最小 environment contract，共享 server core 根据该 contract 与 GGUF PolicySpec 做权威 compatibility check。server 不查找 simulator 原始 observation 字段，也不执行 controller action 转换；client 不复制 server 的 compatibility 规则。

不负责：

- state/action normalization。
- 模型 padding 和真实维度裁剪。
- 模型 prompt template。
- 根据 architecture 选择不同逻辑。
- 创建仿真环境、推进 `env.step()` 或判断任务成功。

### 5.4 Common Policy I/O

`src/policy/` 提供由 PolicySpec 驱动、无状态且可独立测试的公共函数：

- 对命名图像进行存在性、唯一性和角色顺序校验。
- crop、resize、组合、layout 和数值范围转换。
- 对已经按 `StateSpec.fields` 排序的 raw state 做维度校验、padding、mask 和 normalization。
- action 模型维度裁剪、mask、反归一化，以及必要时基于同一次请求 raw state 的 checkpoint 表示恢复。

这些函数不是 `PolicyProcessor` 多态对象，也不定义所有模型都必须采用的 `PreparedInputs` 或 `CorePrediction`。每个具体 `SessionImpl::predict()` 根据自身计算图调用所需函数，并可以在保持相同语义和测试 fixture 的前提下使用 CUDA fused 实现。至少两个模型真正共享的处理才进入公共层；只属于一个 architecture 的 patchify、VAE、token 排列或 scheduler 保留在该模型目录。

`src/policy/image_ops.*` 不能依赖 `GgufReader` 或 GGUF key 名称。GGUF 解析、enum/string 转换和 metadata 错误定位属于 PolicySpec loader；image ops 只执行已经验证的 `ImageSpec`。这样同一 CPU reference 可以直接从测试 fixture 构造 `ImageSpec`，也可以被未来其他 artifact/transport 复用，且不会在每次 `predict()` 中重复解析 metadata。

action decode 函数显式接收 normalized/model-space action、同一次请求的 raw state、`ActionSpec` 和 action statistics。具体 session 必须保留 raw state，并另外产生供模型使用的 normalized/padded state；恢复阶段不能错误地使用 normalized 或 padded state。默认顺序为 `trim real dimension -> unnormalize -> recover representation`，normalization clip 的精确位置在 normalization 公式冻结时统一确定。

语言处理仍位于 serving/eval 与 architecture session 的输入边界：

- `CanonicalObservation` 的 wire contract 保留原始 instruction；request bridge 可以应用 PolicySpec prompt template，并只使用当前 server/model 进程已经具备且经过该 profile 验证的 language path 构造 `wam::Inputs`；
- C++ session 只验证 token/embedding 模式、shape 和 mask；token 输入由 architecture session 的 artifact 内部 text encoder 处理，支持的 embedding 输入可以显式旁路该 encoder。0.5 不实现通用 C++ tokenizer。
- 0.5 不实现通用 external-tokenizer manager，也不规定外部 tokenizer 的 CLI 路径、自动下载、revision 解析、文件校验或跨容器挂载方式。若某个纵向 profile 无法在不依赖该未决机制的情况下构造合法 language input，server 必须在启动时返回 `Unsupported`，不能在请求期临时猜测。

transport 把 WebSocket binary frame 中的 Protobuf/JPEG/PNG payload 解码为 `ImageView`，但不得执行 checkpoint 相关的 resize、composition 或 normalization。

当前 `wam::Inputs.state` 是平坦 tensor，无法在 C++ 中按字段名重排。因此 0.5 不引入具名 state API：observation adapter 负责语义转换并按 `StateSpec.fields` 排序，具体 session 调用公共 state 函数完成后续校验、padding 和 normalization。如果未来把 state 改成具名字段，再单独评估是否把字段装配迁入 C++。

### 5.5 Model Registry 和 Runtime Dispatch

负责：

- 从 `general.architecture` 识别模型。
- 查询 registry 并创建对应 model factory。
- 在模型分配大块显存前加载并校验 `PolicySpec`。
- 管理 Model/Session 生命周期、统一错误、capability 和统计信息。
- 将公共 `Inputs` 交给 policy session，不理解环境名称；environment contract 只存在于 C++ runtime 之外的 server 启动/兼容层。

### 5.6 Architecture Model/Session

每个 architecture 继续通过现有 `ModelImpl`/`SessionImpl` 边界实现完整的一次预测：

- `gwp05`：VAE、UMT5、MoT、prefix/cache、flow matching action denoise。
- `fastwam`：Wan VAE、text encoder、proprio encoder、video/action experts、MoT、action scheduler。

`SessionImpl::predict()` 负责组织“公共输入函数 -> architecture 私有计算 -> 公共 action decode”的顺序，并直接返回 `Prediction`。模型可以定义自己的内部 prepared tensor 和 normalized action 类型，但这些类型不进入公共 runtime 接口，也不要求其他 architecture 复用。session 不读取仿真 observation 字典，不处理 `env.step()`，也不根据 profile/environment 名称做分支。

### 5.7 ActionChunkExecutor

0.5 只实现 simulator side 当前必需的最小 chunk 执行状态：

- action queue；
- `execute_steps`：当前 chunk 最多执行多少行，取值范围为 `[1, horizon]`；
- queue 为空或达到 `execute_steps` 后把控制权交还现有环境循环，由其重新观测和请求推理；
- episode reset 时清理队列。

`ActionChunkExecutor` 可以先用 `reset()/push(chunk, execute_steps)/next_action()/empty()` 这样的小型类或等价函数实现。action ensemble、gripper debounce/hold 和更复杂的 replan policy 等真实需求冻结后再作为组合式控制策略增加，不预先塞入 0.5 基类。`execute_steps` 放在 eval/deployment 配置中，不进入 PolicySpec；C++ runtime 和通用 model server 不持有 action queue。

### 5.8 Simulator Integration

各环境已有入口负责：

- 创建和关闭 simulator/environment；
- reset、initial state、dummy warmup step、`env.step()`、done/success 和视频记录；
- 把 raw observation 交给 adapter 或通过 transport 发送给远端 inference server；
- 执行 ActionChunkExecutor 和 action adapter 给出的 controller command。

0.5 不建立统一 `SimulatorRunner` 基类。RoboTwin、LIBERO 和 LIBERO-X 保留各自远端 client 入口；client 内部完成 CanonicalObservation、单次 payload validation 和 ActionChunkExecutor 所需的最小逻辑，environment/PolicySpec compatibility 由对应 server 入口和共享 server core 判断。只有出现第二个真正相同的实现时才抽取公共生命周期函数。

### 5.9 Serving 和 Transport

负责：

- C ABI、Protobuf message 编解码和 WebSocket binary transport bridge。
- model info、session、predict、reset 和 close。
- 每条 WebSocket 连接创建并独占一个 model session；连接内 predict 串行执行，reset/close 只作用于该连接的 session。
- 环境专用 server 入口声明 environment id/contract，共享 server core 在模型加载后和握手时执行权威 compatibility check。
- 启动握手、最大消息限制、序列化和错误映射。
- 接收 simulator client 已完成环境语义转换的 `CanonicalObservation`，交给 request bridge 构造 `wam::Inputs`。

Transport 不进行图像/state/action 数学处理。0.5 固定使用 WebSocket binary frame 承载 Protobuf payload，不同时维护 gRPC/msgpack 等替代 wire 实现。

### 5.10 处理所有权

| 处理内容 | 唯一所有者 |
| --- | --- |
| 仿真 observation 原始 key 查找 | observation adapter |
| quaternion/axis-angle 等仿真状态语义转换 | observation adapter |
| 相机原始 key 到逻辑 image role 的映射 | observation adapter |
| 按 `StateSpec.fields` 装配 raw state vector | observation adapter |
| prompt template 和已受支持 language path 的 token/embedding 构造 | Serving request bridge / architecture session |
| wire/JPEG/PNG 解码为 `ImageView` | Serving/Transport |
| environment 与 PolicySpec 的兼容性决策 | 对应环境 server 入口提供 contract，共享 server core 执行检查 |
| 图像 crop、resize、组合和数值范围 | 具体 SessionImpl 调用的公共 policy image 函数 |
| state 维度校验、padding 和 normalization | 具体 SessionImpl 调用的公共 policy state 函数 |
| 模型 graph、scheduler、cache 和 denoise | architecture SessionImpl 私有实现 |
| action mask、裁剪、反归一化和基于请求 raw state 的 checkpoint 表示恢复 | 具体 SessionImpl 调用的公共 policy action 函数 |
| PolicyAction 到 controller frame/编码/夹爪符号 | action adapter |
| action queue 和 execute_steps | ActionChunkExecutor（simulator side） |
| environment 创建、reset、step、done/success | 各环境已有 runner/client 入口 |
| 序列化、连接、请求路由和 wire error | Serving/Transport |

同一处理不能在两个模块重复。例如 action adapter 不得再次反归一化 action；未来的跨时间 gripper filter 也不能重新解释 checkpoint 的 gripper normalization。公共 policy 函数负责语义，具体 session 负责调用顺序和 backend 优化，二者不能各自实现一套不同公式。

## 6. PolicySpec 设计

### 6.1 建议的 GGUF metadata

通用字段使用 `wam.*` 前缀，模型 geometry 使用 architecture 前缀：

```text
general.architecture = "fastwam"

wam.artifact_schema_version = 2  # proposed
wam.policy.profile = "libero_panda_2cam_eef"
wam.policy.embodiment = "panda"
wam.policy.training_dataset = "libero"
wam.policy.checkpoint_revision = "..."

wam.input.image.roles = ["scene", "wrist"]

wam.input.image.scene.target_height = 224
wam.input.image.scene.target_width = 224
wam.input.image.scene.resize_mode = "cover_center_crop"
wam.input.image.scene.interpolation = "bilinear"
wam.input.image.scene.antialias = true
wam.input.image.wrist.target_height = 224
wam.input.image.wrist.target_width = 224
wam.input.image.wrist.resize_mode = "cover_center_crop"
wam.input.image.wrist.interpolation = "bilinear"
wam.input.image.wrist.antialias = true

wam.input.image.composition.kind = "canvas"
wam.input.image.composition.height = 224
wam.input.image.composition.width = 448
wam.input.image.composition.scene.rect = [0, 0, 224, 224]  # x, y, width, height
wam.input.image.composition.wrist.rect = [224, 0, 224, 224]
wam.input.image.color_space = "rgb"
wam.input.image.pixel_range = "minus_one_to_one"
wam.input.image.tensor_layout = "chw"

wam.input.state.real_dim = 8
wam.input.state.model_dim = 8
wam.input.state.pad_value = 0.0
wam.input.state.fields = [
  "eef.x", "eef.y", "eef.z",
  "eef.rx", "eef.ry", "eef.rz",
  "gripper.left", "gripper.right"
]

wam.output.action.horizon = 32
wam.output.action.real_dim = 7
wam.output.action.model_dim = 7
wam.output.action.fields = [
  "eef.dx", "eef.dy", "eef.dz",
  "eef.drx", "eef.dry", "eef.drz",
  "gripper.command"
]
wam.output.action.representation = "eef_delta_pose"
wam.output.action.frame = "robot_base"
wam.output.action.gripper = "continuous"
wam.output.action.recovery.kind = "identity"

wam.normalization.state.kind = "min_max"
wam.normalization.action.kind = "min_max"
wam.normalization.epsilon = 1e-6

```

`wam.input.image.roles` 是有序 view 记录的唯一索引；每个 role 的 transform 使用 role 命名的子 key 表达，因此 count 和顺序直接由该数组推导，不再保存 `image.count` 或 `image.0.role`。role 名称必须满足冻结后的字符规则，loader 必须拒绝重复或缺少对应 transform 的条目。

RoboTwin FastWAM profile 可以使用：

```text
wam.input.image.roles = ["scene", "left_wrist", "right_wrist"]
wam.input.image.scene.target_height = 256
wam.input.image.scene.target_width = 320
wam.input.image.scene.resize_mode = "stretch"
wam.input.image.scene.interpolation = "bilinear"
wam.input.image.scene.antialias = true
wam.input.image.left_wrist.target_height = 128
wam.input.image.left_wrist.target_width = 160
wam.input.image.left_wrist.resize_mode = "stretch"
wam.input.image.left_wrist.interpolation = "bilinear"
wam.input.image.left_wrist.antialias = true
wam.input.image.right_wrist.target_height = 128
wam.input.image.right_wrist.target_width = 160
wam.input.image.right_wrist.resize_mode = "stretch"
wam.input.image.right_wrist.interpolation = "bilinear"
wam.input.image.right_wrist.antialias = true
wam.input.image.composition.kind = "canvas"
wam.input.image.composition.height = 384
wam.input.image.composition.width = 320
wam.input.image.composition.scene.rect = [0, 0, 320, 256]
wam.input.image.composition.left_wrist.rect = [0, 256, 160, 128]
wam.input.image.composition.right_wrist.rect = [160, 256, 160, 128]
wam.input.state.real_dim = 14
wam.output.action.real_dim = 14
wam.output.action.representation = "joint_position"
wam.normalization.state.kind = "z_score"
wam.normalization.action.kind = "z_score"
```

当前 GWP-0.5 RoboTwin profile 还需要显式保存 mixed delta/absolute action 的恢复规则：

```text
wam.output.action.representation = "joint_position"
wam.output.action.recovery.kind = "add_current_state"
wam.output.action.recovery.reference_state_indices = [
  0, 1, 2, 3, 4, 5, -1,
  7, 8, 9, 10, 11, 12, -1
]
```

这里 `reference_state_indices[d] >= 0` 表示反归一化后的 action 第 `d` 维需要加上 `raw_state[index]`，`-1` 表示保持绝对值。不得再额外保存一份表达相同信息的 `delta_mask`。上述 key 拼写仍在 Gate B 冻结，但恢复种类、单一索引映射和处理所有权已经确定。

上述名称是 0.5 设计建议，最终编码前需要冻结字段名和枚举值。不能把完整行为压缩为一个 `environment=robotwin` 字段。

### 6.2 Normalization tensor

统计量继续作为 F32 tensor 保存，不放入运行时外部 JSON。0.5 至少支持：

- `z_score`：mean/std；
- `min_max` 或 quantile min/max：q01/q99；
- 可选逐维 normalization mask；
- 当 checkpoint 确实使用时支持逐时间步统计。

建议使用稳定通用名称：

```text
wam.norm.state.mean
wam.norm.state.std
wam.norm.state.q01
wam.norm.state.q99
wam.norm.action.mean
wam.norm.action.std
wam.norm.action.q01
wam.norm.action.q99
wam.norm.action.mask
```

模型私有旧名称可以在 artifact schema 迁移期读取，但 converter 只写新名称。加载时必须校验统计量 shape 与 `real_dim`/`model_dim` 的关系，缺失统计量不能静默退化为 identity。

### 6.3 图像角色与组合

公共输入继续使用命名图像。请求顺序不决定模型顺序，具体 session 调用公共 image 函数，按 `PolicySpec.image.roles` 查找和排序。GGUF 保存 checkpoint 要求的图像处理配置，而不是运行时图像数据；同一个环境的 GWP-0.5 和 FastWAM checkpoint 可以声明不同 transform，adapter 仍然只提交相同的 canonical image roles。

加载和执行链固定为：

```text
GGUF metadata
    -> PolicySpec loader 解析并校验 ImageSpec（model load，只执行一次）
    -> SessionImpl::predict()
    -> 公共 policy image functions（每次请求）
    -> architecture engine 需要的 image tensor
```

0.5 的 per-view resize mode 冻结为以下最小集合：

| mode | 语义 |
| --- | --- |
| `none` | 不改变几何尺寸；若 profile 声明目标尺寸，则输入必须严格匹配 |
| `stretch` | 宽、高独立缩放，直接得到 `target_width x target_height`；FastWAM RoboTwin 使用此模式 |
| `cover_center_crop` | 使用 `scale=max(target_width/source_width, target_height/source_height)` 等比缩放到完全覆盖目标区域，再做整数中心裁剪；GWP-0.5 现有 RoboTwin 路径以及 FastWAM LIBERO/LIBERO-X 使用此模式 |

`cover_center_crop` 的 resized width/height 使用与 Python `round()` 一致的正数 ties-to-even 规则，中心裁剪的 left/top 使用非负整数 floor offset。interpolation 和 antialias 是 transform 的显式字段；同一 artifact schema 下每个枚举必须对应稳定的 CPU reference 像素语义，不能由当前使用 PIL、torchvision、OpenCV 或 CUDA 自动决定。bilinear 的采样坐标、边界和 antialias 精确行为在 Gate B 前由 Phase 0 fixture 冻结；优化实现必须逐像素对齐该 reference。

transport 或直接调用边界先把 wire/JPEG/PNG/BGR 等原始表示转换成 canonical RGB `ImageView`。在此基础上，一次公共 policy 图像处理的规范顺序为：

```text
validate/resolve named roles
    -> per-role resize or crop
    -> optional canvas composition
    -> model pixel-range conversion
    -> tensor layout conversion
```

`pixel_range` 表示交给模型 engine 的最终数值范围，不表示 RPC payload 的输入编码。0.5 的 GWP-0.5/FastWAM profile 使用 RGB 和 `minus_one_to_one`；以后适配需要 per-channel mean/std 的模型时再基于真实 checkpoint 扩展 image value transform，不能把 mean/std 偷藏在 architecture 分支中。

0.5 不为 `horizontal`、`vertical`、`top_with_two_bottom` 分别编写带环境含义的分支。公共 image 函数只需要两种 composition：

| kind | 语义 |
| --- | --- |
| `none` | 每个逻辑视图经过自己的 transform 后仍作为独立 tensor 交给模型 |
| `canvas` | 创建固定大小的输出 canvas，按 placement 把变换后的逻辑视图写入矩形区域 |

每个 `canvas` placement 用 role 命名的 key 显式保存 `x, y, width, height`。loader 必须校验 role 属于唯一的有序 view 记录、矩形在 canvas 边界内、placement 不发生非预期重叠，并校验每个源视图 transform 的输出尺寸与目标矩形一致。双视图横拼、RoboTwin 主视图在上/双腕部在下，以及 GWP-0.5 当前或未来 checkpoint 的 mosaic 都使用同一个数据驱动的 canvas primitive。

composition 只负责几何组合；模型特有 patchify、VAE latent 编码和视觉 token 排列仍属于对应 architecture session。若 CUDA 路径融合 resize/compose/patchify，必须与公共 CPU reference 具有相同像素语义。

环境原始 image key、相机到逻辑 role 的映射，以及由 simulator 原始输出约定造成的 flip/rotation/BGR-to-RGB 等 canonicalization 仍属于 observation adapter。transport 只负责 wire/JPEG/PNG 解码。二者都不得执行 checkpoint 声明的 resize、crop、composition、pixel-range 或 tensor-layout 转换。

### 6.4 Language contract

`PolicySpec` 需要声明：

- 接受 token、embedding 或二者之一；
- prompt template；
- token/embedding shape、最大长度和 mask 规则；
- text encoder 是否在 artifact 中；
- padding side、truncation side、special token 和 attention mask 规则。

环境 adapter 只提供原始 instruction，wire request 不携带 tokenizer 路径或 tokenizer 文件。0.5 暂不闭合 GGUF 外部 tokenizer 的部署方案：不定义 family/revision 到本地文件的解析规则，不自动下载，不定义容器挂载参数，也不计算 tokenizer hash。现有 `LanguageSpec.tokenizer_family/tokenizer_revision` 可以在开发期保留为 provenance/draft 字段，但不能作为已经实现的资源解析能力或 0.5 profile 可用性的保证。

0.5 只支持能够由当前 server/model 部署中已经存在的 language path 构造合法 token/embedding 的纵向 profile。具体 profile 使用 artifact 内部能力、已嵌入资源、固定输入或现有已验证集成路径，需要在该 profile 的 fixture 中说明；依赖尚未实现的 external-tokenizer manager 的 profile 必须启动失败。外部 tokenizer 的资源格式、部署参数和生命周期留到后续版本单独设计，不阻塞当前 C++ 框架、图像/state/action 和 RPC 骨架。

`LanguageSpec.input_mode` 描述 artifact 支持的输入契约；`ModelOptions.language_mode` 描述本次加载选择的运行模式。`LanguageRuntimeMode` 包含 `automatic`、`tokens` 和 `external_embedding`：`automatic` 只允许作为加载请求，成功加载后的 `ModelInfo.language_mode` 必须记录解析得到的具体模式。`tokens` 由 architecture session 使用 artifact 内部的语言编码能力处理 token；`external_embedding` 要求调用方传入与 LanguageSpec 匹配的 embedding，并旁路内部 text encoder。

固定 prompt 是与输入模式正交的加载优化，通过 `ModelOptions.fixed_prompt` 是否存在表达，不再作为 `LanguageRuntimeMode` 的枚举值。只有当前部署已经具备可验证的编码路径时 runtime 才能编码并缓存该 prompt；启用后 `predict()` 不得再接受另一个冲突的语言输入。text encoder 是否仍然加载不使用单独的 `ModelInfo.text_encoder_resident` 布尔值，而由 `ModelInfo.runtime_components` 中对应组件的 `loaded`、`device_bytes` 和加载/卸载耗时表达。

### 6.5 Optional runtime inputs

0.5 的 GWP-0.5/FastWAM PolicySpec 不增加通用 `InferenceInputSpec`。显式 action noise 是调试和数值对齐所需的可选 runtime 能力，通过 `ModelInfo.capabilities.explicit_action_noise` 暴露。调用方通过 `Inputs.action_noise` 提供 little-endian F32 `[horizon, model_action_dim]`；shape 由 ActionSpec 推导，不在 GGUF 中重复保存。0.5 batch inference 尚未启用，因此公共请求不接受额外 batch 维。

`Inputs.action_noise` 为空时，由对应 session 使用 `SessionOptions.random_seed` 初始化的 session-owned RNG 生成。连续预测推进 RNG；显式 action noise 不推进内部 RNG；`session_reset()` 同时重置模型 cache 和 RNG 到 session 初始 seed。seed `0` 是合法的确定性 seed，不表示从系统获取随机 seed。不同随机数库即使 seed 相同也不保证产生相同序列，因此 PyTorch/C++ 数值对齐必须传入相同的显式 action noise，不能依赖 seed 对齐。

`action_noise` 明确且只表示 action denoise 的初始噪声。0.5 的 FastWAM 公共推理模式只支持 action-only/action-noise 路径，不公开 video/world latent noise，不返回 video/world 结果，也不把其他噪声复用或拼接到 `action_noise`。要求调用方提供 video/world noise 或以 joint/video 作为公共输出的 FastWAM 路径必须在加载或 capability check 阶段明确返回 `Unsupported`。这不禁止 FastWAM engine 在 action-only 推理内部执行 checkpoint 数学所必需的 vision/video latent 计算；限制的是公共 runtime 输入输出契约。

当 `Capabilities.explicit_action_noise == false` 时，非空 `Inputs.action_noise` 必须被拒绝。0.5 不增加通用 `noise` alias，也不为未来 video/world 能力预留不明确的 shape；若后续版本正式支持其他随机输入，必须增加语义和 shape 独立的新字段及 capability。

history、temporal image、previous action 和 batch contract 只有在目标 checkpoint 确实要求调用方提供时才扩展。architecture 私有 denoise step、flow scheduler、timestep 和 cache geometry 始终不进入 PolicySpec。

### 6.6 Action contract

ActionSpec 至少需要描述：

- `[horizon, real_action_dim]` 公共输出 shape；
- padded/model action dimension；
- 每一维字段含义；
- `joint_position`、`eef_delta_pose`、`eef_absolute_pose` 等表示；
- `world`、`robot_base`、`eef`、`controller` 等 frame；
- gripper 的连续/离散编码；
- normalization 和 clip；
- 从 core action 到 `PolicyAction` 的确定性恢复规则。

0.5 的最小恢复类型为：

```cpp
enum class ActionRecoveryKind {
    identity = 0,
    add_current_state,
};

struct ActionRecoverySpec {
    ActionRecoveryKind kind = ActionRecoveryKind::identity;
    std::vector<std::int32_t> reference_state_indices;
};
```

新 artifact 必须显式写入 recovery kind，不能因字段缺失静默猜测为 `identity`；只有受版本约束的 legacy migration path 可以在已知旧 profile 语义时显式构造 `identity`。`identity` 不需要索引数组；`add_current_state` 要求数组长度严格等于 `real_action_dim`，每个元素只能为 `-1` 或 `[0, StateSpec.real_dim)` 内的索引。loader 必须在模型分配大块显存前完成这些校验，并拒绝未知/missing kind、越界索引、错误数组长度和缺少 raw state 的调用。恢复公式为：

```text
real_action = trim(model_action, real_action_dim)
real_action = unnormalize(real_action, checkpoint_action_stats)
for d in [0, real_action_dim):
    if reference_state_indices[d] >= 0:
        real_action[:, d] += raw_state[reference_state_indices[d]]
```

所有 horizon 行使用同一次 `predict()` 的 raw state。`reference_state_indices` 已经完整决定哪些维度进行 state-relative 恢复，因此不再增加重复的 `delta_mask` 字段。`ActionSpec.representation` 始终描述恢复完成后返回给调用方的最终语义；GWP-0.5 RoboTwin 为 `joint_position`，而直接返回 EEF delta 的 profile 可以使用 `identity + eef_delta_pose`。

不允许再把公共输出固定为 `[48,14]`。GWP-0.5 RoboTwin artifact 可以仍然声明 `[48,14]`，FastWAM LIBERO artifact 可以声明 `[32,7]`，由 ModelInfo 和返回 tensor shape 给调用者发现。`Prediction.action` 始终是具体 session 已调用公共 action decode 后得到的完整 `[horizon, real_action_dim]`，不能返回 padded `model_action_dim` 给环境侧再裁剪。

`horizon` 与模型的 denoise/inference step 数是两个不同概念。`horizon` 进入 ActionSpec；denoise step 属于 architecture 私有 geometry。`execute_steps` 属于 ActionChunkExecutor deployment config；replan 和 ensemble 属于后续控制策略扩展，均不进入 ActionSpec。

## 7. 公共 API 和内部接口改造

### 7.1 `include/wam/types.h`

现状：

- `ImageView` 已经包含 `name`，可以继续作为逻辑相机角色。
- `Prediction.action` 已经是动态 Tensor。
- `ModelInfo` 尚未暴露 checkpoint 的输入输出契约。
- 存在全局 GWP 相机名称常量，公共语义仍偏向当前 RoboTwin checkpoint。

修改：

1. 增加最小 `PolicySpec` 及其子结构：`PolicyIdentity`、`ImageSpec`、`StateSpec`、`LanguageSpec` 和 `ActionSpec`。
2. `ModelInfo` 增加只读 `policy` 字段。
3. 删除公共层对固定三相机和固定 action shape 的承诺。
4. 保留 `ImageView.name`，但名称由 artifact 要求，不由全局常量规定。
5. 为 action representation、frame、image composition kind、resize/interpolation 和 normalization kind 定义稳定 enum；resize 的 0.5 值为 `none/stretch/cover_center_crop`，未知值必须拒绝。
6. 固定 checkpoint 需求由 PolicySpec 的具体字段表达；`Capabilities` 只表达 runtime 支持的可选调用模式，不添加 `supports_libero` 一类字段，也不重复 view/state/action shape。
7. 0.5 保留 `LanguageInput` 的 token/embedding 边界，不强制给 C++ API 增加 raw text；wire 保留 raw instruction，但只允许使用具体 profile 已验证、当前部署已经具备的 language path 构造输入，外部 tokenizer manager 留到后续版本。
8. 将运行时计算精度类型命名为 `ComputePrecision`，支持 `automatic/f32/f16/bf16/fp8/int8` 扩展；它只描述主要计算精度，不等同于 GGUF 权重存储 dtype 或 block quantization 格式。
9. `ModelOptions.compute_precision` 可以请求 `automatic`，成功加载后的 `ModelInfo.compute_precision` 必须记录解析出的实际精度；`Capabilities.compute_precisions` 列出可选计算精度。
10. `ModelOptions.language_mode` 使用 `LanguageRuntimeMode`，固定 prompt 继续由独立的 `fixed_prompt` 表达；`ModelInfo` 只记录解析后的 language mode，不保存重复且位置语义不清的 `text_encoder_resident`。
11. `Stats` 固定记录 preprocess、model、postprocess 和 total；model 的通用可选分解为 vision、text、prefill 和 decode，额外 architecture 阶段通过 `model_timings` 表达。
12. 对 π0.5，vision/text 分别表示相应 encoder，prefill 表示 VLM prefix 推理，decode 表示 action denoise；对 OpenVLA 等非 denoise 模型，decode 可以表示其直接 action generation 阶段或保持未采集，不得把普通 action head 伪装成 denoise。`model_milliseconds` 始终表示完整模型阶段，`model_timings` 中的条目是细分项而不是第二个总耗时来源。
13. 将可选输入命名为 `Inputs.action_noise`，将对应 capability 命名为 `Capabilities.explicit_action_noise`；非 diffusion/action-denoise 模型将该 capability 保持为 false，并拒绝非空 action noise。

### 7.2 `include/wam/version.h`

修改：

- runtime semantic version 更新到 `0.5.0`。
- GGUF `PolicySpec` 引入不兼容 artifact 变化时提升 artifact schema version。
- ModelInfo 的 Proto 字段采用 additive 方式；只有破坏 wire compatibility 时才提升 protocol major。
- C ABI 是否提升取决于 `ModelOptions` 或 ABI struct 是否变化，不能仅因为 C++ 类型增加而机械提升。
- 在细节冻结前，除 runtime `0.5.0` 外，其余版本号先标记待定。

### 7.3 `src/model_internal.h`

现状：`SessionImpl::predict()` 已经是足够小且稳定的 architecture 多态边界；问题是当前 GWP 实现内部的输入、模型计算和输出语义尚未通过可测试函数明确区分。

修改：

- 保留 `SessionImpl::predict(const Inputs&) -> Prediction` 和 `reset()`，不再增加 `PolicyProcessor` 或 `ModelCoreSession` 多态层。
- `ModelImpl` 持有不可变 `PolicySpec` 和共享模型资源；model-specific cache、RNG 和 episode 状态仍由对应 session 拥有。
- 每个 architecture 的 `SessionImpl::predict()` 显式组织公共 `policy_io` 函数与私有计算步骤，并直接产生完成 decode 的 `Prediction`。
- architecture 可以在自己的目录定义 private `PreparedInputs`/`CorePrediction` 等局部类型，但不放入 `model_internal.h`，不成为其他模型必须适配的统一中间格式。
- 输入校验、模型计算和 action decode 分别有独立函数/fixture，使边界可见并可测试，而不是通过新增虚接口体现。

这样保持与 vla.cpp 接近的单一 `predict()` 核心调用，同时避免把共享 resize/normalization/action decode 复制进每个模型。公共层不建立任意 operator plugin 系统，也不拥有模型 backend buffer。

### 7.4 `src/model.cpp` 和 `src/model_registry.*`

修改：

1. `model_load()` 打开 GGUF 后，先解析通用 schema 和 `PolicySpec`。
2. 在申请模型大块 backend buffer 前完成 profile、normalization tensor 和基本 shape 校验。
3. 将已经解析的 `PolicySpec` 与 `GgufReader` 一起传给 factory。
4. `register_builtin_models()` 增加 `fastwam` factory。
5. `src/model.cpp` 继续只负责 architecture detection、registry dispatch 和生命周期，不能增加 profile/environment switch。
6. `model_info()` 返回完整 PolicySpec，用于 server 启动兼容性检查。

### 7.5 新增 `src/policy/`

建议增加：

```text
src/policy/policy_spec.h
src/policy/policy_spec.cpp
src/policy/image_ops.h
src/policy/image_ops.cpp
src/policy/state_ops.h
src/policy/state_ops.cpp
src/policy/action_ops.h
src/policy/action_ops.cpp
```

职责：

- 从 GGUF 解析和验证通用 PolicySpec。
- 提供命名视图解析、resize/crop/compose、normalization 和 action decode 的无状态函数；action decode 显式接收同一次请求的 raw state，不从 session cache 或 simulator 获取 state。
- 提供 CPU reference 实现；CUDA fused 实现不能改变语义。
- 报错必须带具体字段，例如缺少 `left_wrist`、state dim 不匹配、stats shape 错误。
- 不定义通用 `PolicyProcessor`、`PreparedInputs`、`CorePrediction` 或 architecture plugin 接口。

## 8. GWP-0.5 改造

### 8.1 Artifact

当前 `src/models/gwp05/artifact.cpp` 已读取 `real_state_dim`、`real_action_dim`、`num_views`、image geometry、action chunk 和 normalization tensor，但显式拒绝 `num_views != 3`。当前 RoboTwin GWP VAE 还包含固定三视图的 mosaic/patchify 语义，不能在 PolicySpec 中错误标记为简单的 `separate`。

修改：

- converter 把 view/state/action 等公共契约迁移到 PolicySpec；只有 VAE、MoT、scheduler、latent/cache 等模型私有 geometry 继续使用 `gwp05.*`。
- `artifact.cpp` 验证 GWP 私有 tensor shape 与 PolicySpec 一致。
- normalization 改用通用 tensor 名称，迁移期可以读取旧名称。
- 去掉“全框架固定三视图”的假设，但保留“当前 artifact 实际要求 N 个模型视图”的严格校验。

是否能让 GWP-0.5 使用两视图必须以 upstream checkpoint 为准：

- 如果 checkpoint 和图结构原生支持动态视图，按 `PolicySpec.image.roles.size()` 建图。
- 如果 checkpoint 固定三个视觉 slot，只能在 profile 明确声明且 upstream 训练采用相同 mask/占位语义时补空视图。
- 禁止为了通过 LIBERO 接口而私自补零图像；这会改变模型数学语义且无法作为正确适配。

### 8.2 Inputs 和 semantics

当前 `src/models/gwp05/semantics.cpp` 固定三个相机名称和顺序。

修改：

- 将相机角色和顺序改为来自 PolicySpec。
- `inputs.cpp` 负责在 GWP session 内调用公共 image/state 函数，并把结果转换为 GWP engine 输入。
- state padding/normalization 和 action trim/unnormalization/recovery 迁移到通用 policy 函数；observation adapter 先按 `StateSpec.fields` 生成有序 raw state，GWP session 同时保留 raw state 和派生出的 model state。
- 当前 RoboTwin profile 在 action 反归一化后按 `reference_state_indices=[0,1,2,3,4,5,-1,7,8,9,10,11,12,-1]` 加回同一次请求的 raw state；该规则来自 PolicySpec，不在 GWP engine、server 或 RoboTwin client 中硬编码。
- token 顺序、RoPE、MoT mask、scheduler 和 cache shape 等 GWP 专属语义继续保留在 `gwp05/semantics.*`。

### 8.3 Model 和 engine

修改：

- `model.cpp` 创建单一 GWP `SessionImpl`；session 内部调用公共 policy 函数和 GWP engine。
- engine 从 geometry 获取真实 view/token 数，不读取环境 profile 名称。
- 公共 action 输出 shape 使用 PolicySpec 的 horizon 和 real action dim。
- 保持现有 F32/BF16 路径、cache、scheduler 和已验证数值语义不变。
- 首先使用现有 RoboTwin GGUF 做无行为重构，证明改造前后 bitwise 或容差内一致，再添加其他 profile。

## 9. FastWAM 新增实现

### 9.1 目录和 factory

新增：

```text
src/models/fastwam/artifact.h
src/models/fastwam/artifact.cpp
src/models/fastwam/model.h
src/models/fastwam/model.cpp
src/models/fastwam/semantics.h
src/models/fastwam/semantics.cpp
src/models/fastwam/engine/
```

并在 `src/arch.h`、`src/model_registry.cpp`、`cmake/WamModels.cmake` 和 `cmake/WamOptions.cmake` 中注册 `fastwam` 与 `WAM_BUILD_FASTWAM`。

### 9.2 GGUF 组成

FastWAM converter 需要根据实际 checkpoint 收集：

- video expert；
- action expert/ActionDiT；
- MoT mixed-attention 参数；
- Wan VAE；
- text encoder 和 prompt 相关配置；
- proprio encoder；
- video/action scheduler 参数；
- PolicySpec 和 dataset normalization statistics。

0.5 的 FastWAM 必需且唯一的 policy 输出是 action chunk，唯一公开可控的扩散初始噪声是 `Inputs.action_noise`。video/world 输出、video/world noise 以及 joint/video 公共推理模式不进入 0.5；相关路径必须显式 unsupported，不能通过 `Prediction.auxiliary` 或含义不明确的 `noise` 字段半公开。

### 9.3 LIBERO 和 LIBERO-X profile

根据当前 FastWAM 配置，至少需要支持：

- 两个相机角色：`scene`、`wrist`；
- 每视图 `224x224`；
- 横向拼接为 `224x448`；
- 8 维 state；
- 7 维 action；
- 前 6 维为 EEF delta，gripper 为非 delta；
- LIBERO checkpoint 使用 min/max，LIBERO-X checkpoint 可能使用 z-score，必须由各自 artifact 声明，不能根据环境名猜测。

LIBERO 和 LIBERO-X 的图像翻转、原始 key、reset 和夹爪 controller 行为保留在各自 adapter。

### 9.4 RoboTwin profile

至少需要支持：

- `scene`、`left_wrist`、`right_wrist` 三个角色；
- 主视图 resize 到 `256x320`；
- 两个腕部视图分别 resize 到 `128x160`；
- 通过 `canvas=384x320` 和三个显式 placement 组合，而不是使用带环境含义的 `top_with_two_bottom` 分支；
- 14 维 state/action；
- z-score normalization；
- `joint_position`/`qpos` action 语义。

上述组合属于 FastWAM checkpoint 输入处理，因此由 FastWAM session 调用 PolicySpec 驱动的公共 image 函数完成，而不是由 RoboTwin adapter 拼成一张匿名图像。

### 9.5 数值对齐

FastWAM 实现必须分阶段与 PyTorch 对齐：

1. 单视图 resize/crop 和最终 composite image。
2. state normalization 和 proprio token。
3. VAE latent。
4. text embedding。
5. 首层/末层 MoT 中间 tensor。
6. 固定 action noise 下的 action denoise trajectory。
7. normalized action。
8. 反归一化后的公共 PolicyAction。

只有最终 action 接近但中间过程无法解释，不能作为转换正确的充分证据。

## 10. 仿真环境适配层

### 10.1 目标目录

仿照 vla.cpp，把 simulator checkout、环境 client 和 server 启动入口集中到 `eval/sim/`。`eval/` 后续可以增加与 `sim/` 平级的真机目录，而不改变 model server 和 runtime：

```text
eval/
  sim/
    RoboTwin/                 # upstream checkout, ignored by wam.cpp git
    LIBERO/                   # upstream checkout, ignored by wam.cpp git
    LIBERO-X/                 # upstream checkout, ignored by wam.cpp git
    run_robotwin_client.py
    run_robotwin_server.py
    run_libero_client.py
    run_libero_server.py
    run_liberox_client.py
    run_liberox_server.py
    setup_robotwin.sh
    setup_libero.sh
    setup_liberox.sh
  common/
    rpc.py
    server.py
```

三个 `setup_*.sh` 负责 clone 上游仓库、checkout 版本控制中记录的固定 revision 并安装环境依赖。评测过程不能自动跟随上游最新分支执行无约束 `git pull`；更新 simulator 必须显式修改 revision 并重新验证。三个 upstream checkout 通过根 `.gitignore` 排除，不复制进 wam.cpp 的版本历史。

三个 `run_*_server.py` 是共享 `eval/common/server.py` 的环境专用启动入口。每个入口固定声明自己的 environment id 和最小 environment contract，并可以提供默认端口；共享 server core 使用 contract 检查 GGUF PolicySpec。入口不包含 simulator observation parser，不根据模型名称选择计算路径，也不覆盖 GGUF 中的 view/state/action/normalization。切换 GWP-0.5/FastWAM 仍然只替换该环境 server 的 `--model`。

三个 `run_*_client.py` 分别导入对应 upstream simulator，持有环境生命周期、环境语义转换和 action chunk 执行。0.5 不建立 adapter、runner 或 client 基类，也不预先拆分 `request_builder.py`、`action_chunk.py` 等文件；真实重复出现后再从这些入口中抽取小型公共函数。

### 10.2 Environment adaptation contract

每个 `run_*_client.py` 在自身模块内提供以下语义；函数名可以根据上游入口调整，但不得通过 architecture 名称选择不同 parser：

```python
def observation_to_policy_observation(observation, policy_spec): ...
def validate_observation_payload(canonical_observation): ...

def policy_action_to_command(action, policy_spec): ...
def validate_action_payload(policy_action): ...

def reset_observation_state(): ...  # only when stateful
def reset_action_state(): ...       # only when stateful
```

三个 `run_*_server.py` 分别提供一个数据化 environment contract；共享 server core 使用同一函数检查：

- 能否提供所有 image role；
- canonical state fields/layout 和维度；
- 环境能否消费 action representation/frame/layout；
- gripper 语义；

0.5 的最小 contract 可以直接使用不可变 Python 数据，不建立 adapter 基类：

```python
@dataclass(frozen=True)
class ActionConsumerContract:
    real_dim: int
    representation: str
    frame: str
    gripper: str

@dataclass(frozen=True)
class EnvironmentContract:
    environment_id: str
    producible_image_roles: frozenset[str]
    supported_state_fields: tuple[tuple[str, ...], ...]
    supported_actions: tuple[ActionConsumerContract, ...]
```

checker 要求 `PolicySpec.image.roles` 是 `producible_image_roles` 的子集，`StateSpec.fields/real_dim` 精确匹配一个 supported state layout，并且 `ActionSpec` 的 real dimension、representation、frame 和 gripper 组合精确匹配一个 `supported_actions` 条目。contract 不包含 simulator 原始 key、episode/task、checkpoint normalization、image transform 或模型名称；这些分别属于 client、PolicySpec 或 architecture。

server 在模型加载后先根据启动入口选定的 environment contract 检查 ModelInfo/PolicySpec，不兼容则在创建 session 和接受 episode 前失败。client 的第一条 WebSocket 消息还必须声明相同 environment id；id 不一致时握手失败。检查不能只比较 `profile == environment_id`，必须比较上述结构化字段，但 0.5 的权威结果只由 server 返回。

`CanonicalObservation` 是 RPC 的模型无关输入，至少包含命名图像、按 `StateSpec.fields` 排序但未 normalization/padding 的 raw state，以及原始 instruction。client 不执行 checkpoint image resize/composition、state normalization 或 prompt template。client 只验证实际 payload 是否满足握手返回的基本字段/shape，server/request bridge 再根据 PolicySpec 构造输入；这种本地防御性检查不能改变或覆盖 server compatibility 结论。

server 返回的 `Prediction.action` 已经是完整 `[horizon, real_action_dim]` PolicyActionChunk。client 只执行环境 controller 所需的 frame、gripper 和调用格式转换，不重复 checkpoint action decode。

### 10.3 ActionChunkExecutor

0.5 在各 simulator client 中只实现已经确认的队列语义：

```python
class ActionChunkExecutor:
    def reset(self): ...
    def push(self, chunk, execute_steps): ...
    def next_action(self): ...
    def empty(self): ...
```

完整 PolicyActionChunk 通过 transport 返回 simulator side，再由 client 决定逐步执行、丢弃剩余 action 或重新推理。三个 client 保留自身的 reset、observe、step、done/success 和 close 流程，不为接入 wam.cpp 重写成统一基类。只有至少两个真实 client 形成完全相同的队列实现后，才考虑把该小类移入 common；action ensemble、debounce/hold 等尚未冻结的策略不进入基础 RPC 或 model server。

### 10.4 LIBERO client/server

`run_libero_client.py` 负责：

- 从 `eval/sim/LIBERO/` 导入固定 revision 的上游环境。
- 提取 agent/scene 与 wrist 图像并映射逻辑角色。
- 将 EEF position、quaternion 和 gripper qpos 组成 profile 要求的 state；quaternion 到 axis-angle 属于这里。
- 将 EEF action 和 gripper command 转成 LIBERO `env.step()` 接受的格式。
- 处理 LIBERO controller 需要的夹爪符号，但不做 checkpoint action unnormalization。
- 处理 reset、initial state、dummy wait steps、done/success、action queue 和视频记录。

`run_libero_server.py` 使用 `environment_id="libero"` 和 LIBERO environment contract 启动共享 server core。LIBERO 与 model server 可以位于不同 Conda/Docker 环境，通过 host/port 连接。

### 10.5 LIBERO-X client/server

`run_liberox_client.py` 负责：

- 从 `eval/sim/LIBERO-X/` 导入固定 revision 的上游环境。
- 解析 openpi/LIBERO-X 的 `observation/image`、`observation/wrist_image`、`observation/state` 和 prompt。
- 处理 LIBERO-X 原始图像方向、flip 和环境字段差异。
- 将公共 action 转换为 LIBERO-X controller action。
- 把当前 FastWAM LIBERO-X policy 中的 controller gripper sign/range/binarize 放入 action adapter；确有需要时，把跨时间 debounce/hold 作为独立控制 filter 加在 ActionChunkExecutor 之后，不能保留在 FastWAM session。
- 处理 scene group、BDDL/init、episode 上限、action queue、成功判定和视频记录。

`run_liberox_server.py` 使用 `environment_id="liberox"` 和 LIBERO-X environment contract 启动共享 server core。现有 FastWAM `LIBEROX_REMOTE_EVAL.md` 的双容器方式作为迁移参考；client 发送 CanonicalObservation，server 不维护 LIBERO-X/openpi 原始 observation parser。

### 10.6 RoboTwin client/server

`run_robotwin_client.py` 负责：

- 从 `eval/sim/RoboTwin/` 导入固定 revision 的上游环境。
- 兼容当前扁平 key 和嵌套 observation key。
- 映射 head/left wrist/right wrist 图像角色。
- 提取 14 维 `joint_action.vector` 或正式冻结后的 state layout。
- 将 `joint_position` PolicyAction 交给 RoboTwin qpos controller。
- 处理 instruction、task 生命周期、ActionChunkExecutor、qpos command 和 episode 结果。
- reset 控制消息触发 server session reset，并清空 client 的 ActionChunkExecutor/action state。
- 当前逐行执行返回 chunk 中全部 action 的行为对应 `execute_steps=horizon`；后续可以通过 client deployment config 选择更短的 receding-horizon 执行，不修改 GGUF。

`run_robotwin_server.py` 使用 `environment_id="robotwin"` 和 RoboTwin environment contract 启动共享 server core。迁移当前旧版 `eval/robotwin/inference_server_wam.py` 时，RoboTwin 原始 observation parser 和 GWP-specific model 名称不能进入 server core；environment contract 可以声明 RoboTwin adapter 能提供/消费的相机角色与 state/action 语义，但实际 checkpoint shape 仍来自 PolicySpec，不能在 server 中用 `ACTION_SHAPE=(48,14)` 覆盖。

## 11. Serving 和协议改造

0.5 的远程 eval wire contract 冻结为：WebSocket binary frame 承载 Protobuf payload；一条 WebSocket 连接拥有一个 model session；同一连接/session 内所有操作按接收顺序串行执行。PolicySpec 子 message 的最终字段号仍在 Gate B 冻结，但 transport、envelope、连接生命周期和并发语义不再保留多套候选方案。

### 11.1 Wire transport 和连接生命周期

- endpoint 使用 `ws://host:port`；TLS、认证和反向代理不属于本地仿真 0.5 范围。
- client/server 只发送 WebSocket binary message，不接受 text/JSON/msgpack frame。一个完整 binary message 恰好包含一个 serialized Protobuf envelope；底层 WebSocket fragmentation 由库重组，不暴露给应用协议。
- client 建连后的第一条消息必须是 `HelloRequest`。server 检查 protocol major、environment id 以及启动时已经验证的 environment/PolicySpec compatibility；成功后为该连接创建且只创建一个 `wam::Session`，并返回 `HelloResponse(ModelInfo)`。
- 握手成功后只允许 `PredictRequest`、`ResetRequest` 和 `CloseRequest`。每个请求携带连接内唯一且单调递增的 `request_id`，response 原样回传该 id。
- 同一连接最多存在一个 in-flight request。server 不并发调用同一个 session 的 `predict()`，client 必须收到前一个 response 后才能发送下一个请求；违反顺序返回 protocol error 并关闭连接。
- `ResetRequest` 只调用当前连接 session 的 `session_reset()`；只有 reset response 返回后才能继续 predict。`CloseRequest` 或 WebSocket 断开立即销毁该 session。重连创建新 session，不继承 cache、RNG 或 episode 状态。
- 一个 server process 可以共享一个只读 `Model` 并服务多条连接，但每条连接有独立 session，连接之间是否并行由 model/runtime capability 和 server deployment config 决定，不改变单 session 串行语义。
- 握手 protocol/environment 不兼容、非法 envelope 和请求乱序是 fatal error，server 返回 `RpcError(fatal=true)` 后关闭连接。单次 predict 的字段/shape 错误返回带 `request_id` 的 non-fatal error，session 保持可 reset 或继续使用；backend/session 损坏时返回 fatal error。

规范时序为：

```text
connect
  -> HelloRequest(protocol_version, environment_id)
  <- HelloResponse(ModelInfo + PolicySpec)
  -> PredictRequest(request_id=1, CanonicalObservation, optional action_noise)
  <- PredictResponse(request_id=1, Prediction)
  -> ResetRequest(request_id=2)
  <- ResetResponse(request_id=2)
  -> CloseRequest(request_id=3) or WebSocket close
```

### 11.2 `CanonicalObservation` 和 `proto/wam.proto`

0.5 的 envelope 结构冻结为以下语义，实际 `.proto` 必须为新增字段分配稳定且不复用的 field number：

```proto
message ClientEnvelope {
  uint64 request_id = 1;
  oneof payload {
    HelloRequest hello = 10;
    PredictRequest predict = 11;
    ResetRequest reset = 12;
    CloseRequest close = 13;
  }
}

message ServerEnvelope {
  uint64 request_id = 1;
  oneof payload {
    HelloResponse hello = 10;
    PredictResponse predict = 11;
    ResetResponse reset = 12;
    CloseResponse close = 13;
    RpcError error = 14;
  }
}

message HelloRequest {
  uint32 protocol_major = 1;
  uint32 protocol_minor = 2;
  string environment_id = 3;
}

message HelloResponse {
  ModelInfo model_info = 1;
}

enum TensorDType {
  TENSOR_DTYPE_UNSPECIFIED = 0;
  TENSOR_DTYPE_F32 = 1;
  TENSOR_DTYPE_I32 = 2;
  TENSOR_DTYPE_U8 = 3;
}

message Tensor {
  TensorDType dtype = 1;
  repeated uint64 shape = 2;
  bytes data = 3;  // contiguous little-endian row-major bytes
}

enum ImageEncoding {
  IMAGE_ENCODING_UNSPECIFIED = 0;
  IMAGE_ENCODING_RGB_U8 = 1;
  IMAGE_ENCODING_PNG = 2;
  IMAGE_ENCODING_JPEG = 3;
}

message Image {
  string name = 1;
  ImageEncoding encoding = 2;
  uint32 width = 3;
  uint32 height = 4;
  bytes data = 5;
}

message CanonicalObservation {
  repeated Image images = 1;
  Tensor state = 2;
  string instruction = 3;
}

message PredictRequest {
  CanonicalObservation observation = 1;
  Tensor action_noise = 2;  // message presence means explicitly provided
}

message PredictResponse {
  Prediction prediction = 1;
}

message ResetRequest {}
message ResetResponse {}
message CloseRequest {}
message CloseResponse {}

enum RpcErrorCode {
  RPC_ERROR_UNSPECIFIED = 0;
  RPC_ERROR_PROTOCOL = 1;
  RPC_ERROR_INCOMPATIBLE = 2;
  RPC_ERROR_INVALID_ARGUMENT = 3;
  RPC_ERROR_UNSUPPORTED = 4;
  RPC_ERROR_INTERNAL = 5;
}

message RpcError {
  RpcErrorCode code = 1;
  string field = 2;
  string message = 3;
  bool fatal = 4;
}
```

`ResetRequest/Response` 和 `CloseRequest/Response` 是空控制 payload；`RpcError` 至少包含稳定 error code、field、message 和 `fatal`。首个 `HelloRequest` 的 envelope `request_id=0`，握手后的控制/预测 request id 从 1 开始严格递增。

`CanonicalObservation` 的 wire 语义为：

- `images`：命名图像，name 唯一；encoding/width/height/data 由 `Image` 自描述。环境原始 key 已由 client 转成 canonical role，但尚未执行 checkpoint resize/composition/pixel-range/layout。
- `RGB_U8` 的 `data` 必须是无 padding 的紧密 HWC RGB，长度严格等于 `width*height*3`；PNG/JPEG 的 width/height 必须与解码结果一致。0.5 wire 不传 row stride、BGR 或任意像素 layout。
- `state`：little-endian F32、rank-1 `[real_state_dim]`，已经按 `StateSpec.fields` 排列，但尚未 normalization/padding。
- `instruction`：原始 UTF-8 任务文本；不携带 prompt template、tokenizer path、tokenizer 文件或外部资源 locator。
- `action_noise`：可选 little-endian F32、rank-2 `[horizon, model_action_dim]`；缺省时使用 session RNG。capability 为 false 时提供该字段必须失败。
- `Prediction.action`：little-endian F32、rank-2 `[horizon, real_action_dim]` 的完整 PolicyActionChunk。

修改现有 Proto 时还需要：

- 为 `ModelInfo` 添加 `PolicySpec` 及其子 message。
- 保留现有字段号，新增字段只能使用新的编号。
- 保留现有 `artifact_sha256` 字段号但返回空字符串，不能删除、复用或作为客户端启动条件。
- `Prediction.action` 继续通过 Tensor 自描述 shape，不新增固定 action shape 字段。
- `Image.name` 继续承载逻辑角色。
- C++ `Inputs.action_noise` 在 wire request 中同样使用 `action_noise`，不保留含义不明确的通用 `noise` 名称。
- error detail 对缺失角色、维度、representation 和 environment compatibility 使用稳定 field 名称。

### 11.3 `src/serving/protocol_adapter.cpp`

修改：

- 序列化完整 PolicySpec。
- 对新增枚举进行显式映射，未知枚举失败，不能默认转换。
- 保持 request tensor 的 size/overflow/byte order 校验。
- 校验 envelope oneof、连接状态、严格递增 request id 和 CanonicalObservation/action_noise 的 dtype/shape。
- 增加 PolicySpec Proto round-trip fixture。

### 11.4 Python serving bridge

修改：

- `rpc.py` 和动态 Proto 类型能够读写完整 envelope、CanonicalObservation 和 PolicySpec。
- `rpc.py` 实现单一 WebSocket/Protobuf codec 和上述顺序状态机，不增加 JSON/msgpack fallback。
- environment-specific server 入口加载模型后把 environment contract 交给共享 server core；server 在开始监听或接受 episode 前执行 compatibility check，握手返回完整 ModelInfo/PolicySpec 或 fatal error。
- client 只声明 environment id、消费 server 返回的 ModelInfo 并做实际 payload 防御性检查，不维护另一份 environment/PolicySpec compatibility 判定。
- server 握手返回 architecture、profile、checkpoint revision、artifact schema/size、输入角色、state/action spec、runtime/protocol version，不返回或依赖完整文件 hash。
- 删除客户端手工传入 `arch`、action shape、view count 和 stats JSON 的需要。
- backend、precision、device 等 runtime 参数仍可以通过 server CLI 配置；`execute_steps` 由 simulator runner/client 的 deployment config 配置，replan/ensemble 在相应控制策略实现后再定义配置。

### 11.5 启动方式

目标示例：

```bash
python eval/sim/run_robotwin_server.py \
  --library build/libwam_c_api.so \
  --descriptor build/wam.desc \
  --model models/fastwam-robotwin-dual-arm.gguf \
  --backend cuda \
  --precision bf16
```

切换 GWP-0.5 时只替换：

```text
--model models/gwp05-robotwin-dual-arm.gguf
```

如果加载 LIBERO GGUF 启动 RoboTwin server，server 必须在接受第一个 episode 前因相机角色、state layout 或 action representation 不兼容而失败。

## 12. Converter 和检查工具改造

### 12.1 公共 profile 输入

训练 checkpoint 未完整保存部署配置时，converter 可以接受版本控制的 profile YAML/JSON，例如：

```text
configs/policies/gwp05/libero.yaml
configs/policies/gwp05/liberox.yaml
configs/policies/gwp05/robotwin.yaml
configs/policies/fastwam/libero.yaml
configs/policies/fastwam/liberox.yaml
configs/policies/fastwam/robotwin.yaml
```

这些文件只作为转换输入。converter 必须将其规范化内容写入 GGUF，runtime 和 server 不能依赖这些外部配置才能解释模型。

优先从 checkpoint 自带 config 和 dataset statistics 自动提取；只有缺失内容才要求显式 profile。可选 conversion report 记录来源路径或 revision，但 0.5 不计算来源文件 hash。

### 12.2 `convert_gwp05.py`

修改：

- 写入通用 PolicySpec。
- 写入稳定通用 normalization tensor。
- 校验 profile view 数与 checkpoint 图结构能力。
- 为当前 RoboTwin artifact 提供迁移模式并生成改造前后 parity fixture。
- 禁止使用 `--environment` 单个字符串替代完整 profile。

### 12.3 新增 `convert_fastwam.py`

负责：

- 合并 FastWAM 所需 component 和 fine-tuned checkpoint。
- 从 Hydra/model/data 配置解析 geometry、scheduler、相机组合、state/action 和 normalizer。
- 写入 architecture 私有 metadata、通用 PolicySpec 和 stats。
- 检查 model weight 中 proprio/action 维度与 profile 一致。
- 输出 component telemetry、source revision/path 和转换报告，不计算完整 source/artifact hash。

### 12.4 `inspect_gguf.py`

修改：

- 打印规范化 PolicySpec 摘要。
- 验证 image composition 的源/目标尺寸。
- 验证 state/action normalization tensor。
- 验证 action horizon、real/model dim 和模型输出 head。
- 支持 `--check-environment libero|liberox|robotwin` 的静态兼容性检查；该功能调用与 environment-specific server 相同的 contract/checker，不能复制另一套规则。

## 13. 构建和目录调整

建议的 0.5 目标结构：

```text
include/wam/
  types.h
  version.h
  wam.h

src/
  model.cpp
  model_internal.h
  model_registry.*
  policy/
    policy_spec.*
    image_ops.*
    state_ops.*
    action_ops.*
  models/
    common/
    gwp05/
    fastwam/
  serving/

eval/
  sim/
    RoboTwin/                 # ignored upstream checkout
    LIBERO/                   # ignored upstream checkout
    LIBERO-X/                 # ignored upstream checkout
    run_robotwin_client.py
    run_robotwin_server.py
    run_libero_client.py
    run_libero_server.py
    run_liberox_client.py
    run_liberox_server.py
    setup_robotwin.sh
    setup_libero.sh
    setup_liberox.sh
  common/
    rpc.py
    server.py

scripts/
  convert/
    convert_gwp05.py
    convert_fastwam.py
  inspect/
```

`src/models/common/` 只放至少两个模型真正共享的模型数学或加载工具；环境 adaptation 和 action chunk 执行不进入 `wam_core`。`src/policy/` 只放 PolicySpec 与无状态 image/state/action 函数，不增加 processor/plugin 基类。Python `eval/common/` 在 0.5 承载 WebSocket/Protobuf 编解码、连接状态机、environment-contract compatibility 函数和共享 model server core；三个 `run_*_server.py` 提供各自 environment id/contract，环境生命周期与原始 observation/controller 转换仍保留在对应 client 入口。

## 14. 测试和验收

### 14.1 Schema 测试

- 六种目标 profile 的 metadata fixture。
- 缺失字段、未知枚举、错误 view 数、非法 canvas placement、stats shape 和 action head shape 的失败测试。
- converter、inspector 和 runtime 对同一 schema 的一致解释。

### 14.2 Policy I/O 和环境适配测试

- 每个环境冻结一份 raw observation fixture。
- 各 simulator client 输出的命名图像和按 `StateSpec.fields` 排序的 raw state 与 Python reference 对齐。
- resize/crop/compose 的像素对齐。
- min/max、quantile、z-score normalization 对齐。
- action trim/mask/unnormalization 对齐。
- `identity` 与 `add_current_state` action recovery 对齐；覆盖 missing/unknown kind、索引 `-1`、越界索引、错误长度、缺少 raw state，以及所有 horizon 行引用同一请求 raw state 的测试。
- 相机输入乱序不影响结果；缺失和重复角色明确失败。

### 14.3 模型数值测试

对每个可获得 checkpoint 的模型/环境组合执行：

- 固定 prompt、state、image 和 action noise。
- PyTorch 与 C++ 中间 tensor 对齐。
- 最终 normalized action 和 PolicyAction 对齐。
- F32 作为 reference，BF16 使用单独容差。
- refactor 前后的现有 GWP-0.5 RoboTwin fixture 不回归。

### 14.4 Serving 测试

- ClientEnvelope/ServerEnvelope、CanonicalObservation、ModelInfo/PolicySpec Proto round trip。
- C ABI serialized request/response。
- 只接受 WebSocket binary Protobuf message，拒绝 text/JSON/msgpack frame、非法 oneof 和超大 payload。
- 强制 Hello-first、protocol major、environment id 和 request id 单调递增；覆盖环境不匹配和请求乱序的 fatal error。
- server 启动时使用对应 environment contract 做权威兼容性检查；client 不执行第二套 PolicySpec compatibility 判定。
- 一条连接只创建一个 session，同一 session 最多一个 in-flight predict；多连接 session 独立，断线销毁对应 session。
- server/session reset 清空模型 cache；simulator client reset 清空 ActionChunkExecutor queue 和 action adapter/filter 状态，分别测试各自所有者。
- dynamic action shape 不再依赖 `[48,14]` 常量。
- server 始终返回完整 `[horizon, real_action_dim]`，不根据 `execute_steps` 截断 response。
- 连接断开、超大消息、错误 dtype/shape 和 timeout。
- action noise 缺省、显式输入、错误 dtype/shape、非有限值、RNG 连续推进及 reset 后复现。

### 14.5 仿真测试

每个组合至少完成：

1. server/client 启动 smoke test；
2. 一个 episode 的完整推理和 action 执行；
3. 多次 replan 和 reset；
4. action、latency、错误和 artifact identity 日志；
5. 有有效微调 checkpoint 时执行正式成功率评测。

发布报告必须区分：

- runtime parity；
- integration pass rate；
- benchmark success rate；
- checkpoint 来源和是否针对该环境微调。

## 15. 实施顺序

0.5 采用“最小骨架先行、真实纵向路径验证、profile 逐步填充”的方式实施。前面列出的 profile 细节不全部阻塞骨架建设；第一阶段只冻结模块边界、核心数据流、生命周期、错误语义和 action 输出契约。尚未实现的 architecture、profile 或处理函数必须明确返回 `Unsupported`/`NotImplemented`，禁止用补零、identity normalization、默认 action 语义或其他猜测使测试表面通过。

每个阶段都必须保持主分支可构建。框架接口可以在 0.5 开发期根据两次架构审查调整，在第二次审查通过前不承诺稳定 artifact/ABI；公共 GGUF key、Proto 字段和枚举一旦冻结，则按版本兼容规则演进。

### Phase 0：冻结现状

- 固定当前 GWP-0.5 RoboTwin 输入、输出、artifact 和 PyTorch/C++ fixture。
- 在 GWP fixture 中固定 normalized `[48,32]` action、反归一化后的 `[48,14]` mixed action、请求 raw state、reference-state mapping 和最终 `[48,14]` joint-position action，分别验证每个恢复阶段。
- 固定 FastWAM 三套 Python preprocessing、normalization 和 action 后处理 fixture。
- 记录 GWP-0.5 LIBERO/LIBERO-X 权重为训练中，记录 FastWAM 三套可用权重的路径、revision、模型变体和许可证。
- 记录当前 server 协议和启动命令。
- 以 WebSocket binary + Protobuf、连接级 session 和单 session 串行 predict 作为 0.5 新协议 baseline；旧协议只作为迁移输入，不再作为并行目标。

### Phase 1：C++ 框架骨架

- 建立 `src/policy/`、`src/models/fastwam/` 和 registry/factory 目录骨架。
- 引入内部 `PolicySpecDraft` 和 `PolicyActionChunk`；此阶段只保证数据所有权和 action 边界正确，metadata key 的最终拼写后续冻结。
- 保留现有 `ModelImpl`/`SessionImpl::predict()` 多态边界，建立命名视图、state normalization 和 action decode 的最小无状态函数骨架；不增加 `PolicyProcessor`、`ModelCoreSession` 或通用中间 tensor 类型。
- 保持 model/session/cache 生命周期、backend 资源所有权和错误传播边界清晰。
- 为尚未实现的 FastWAM factory/profile 提供显式 unsupported 路径，并在分配大块显存前失败；不得返回伪 action。
- 添加接口构造、registry dispatch、unsupported error 和 session lifecycle 的骨架测试。
- 建立 `eval/sim/` 和 `eval/common/` 的声明骨架及 ignored upstream checkout 路径；所有入口在真实集成前显式 `NotImplemented`，Phase 1 不拉取 simulator 或提供伪远程行为。

### Phase 2：GWP-0.5 RoboTwin 纵向验证

- 从现有 GWP 私有 metadata 构造临时的内部 PolicySpec，作为明确隔离的 legacy migration path。
- 在现有 GWP `SessionImpl::predict()` 内调用公共 image/state/action 函数；GWP engine、cache 和私有 prepared tensor 保持在 `src/models/gwp05/`。
- GWP engine 只返回 normalized/model-space action；session 在返回 `Prediction` 前依次执行 trim、反归一化和 `add_current_state`，并确保使用本次请求的 raw state。server 和 RoboTwin client 删除或禁止重复恢复。
- 先保持现有 GWP 数学、cache、scheduler、F32/BF16 和 CUDA 路径不变，不在本阶段顺带重写 kernel。
- runtime 返回完整 `[48,14]` PolicyActionChunk；RoboTwin server 不再从独立常量重建 action shape。
- 使用 Phase 0 fixture 证明改造前后输出 bitwise 一致或在冻结容差内一致，并检查性能没有不可接受回归。

### Review Gate A：骨架审查

- 核对 observation/action adaptation、serving request bridge、公共 policy 函数、architecture session 和 ActionChunkExecutor 的职责是否仍然唯一。
- 核对公共层没有 RoboTwin/GWP 专用分支，模型层没有环境名称。
- 核对 legacy PolicySpec 构造逻辑被隔离，不能成为新 artifact 的隐式默认值。
- 核对 `SessionImpl::predict()` 仍是唯一 architecture 推理多态入口，没有为了阶段拆分引入新的强制虚接口。
- 只有 GWP-0.5 RoboTwin 无行为回归、错误边界和资源生命周期验证通过后，才确认最小骨架并进入第二种模型验证。

### Phase 3：FastWAM 首个 profile 纵向验证

- 从三套现有权重中选择一个明确的 0.5 FastWAM 变体和 profile，优先完成最容易建立完整 PyTorch fixture 的组合。
- 所选变体必须支持 0.5 的 action-only/action-noise 公共契约；要求外部 video/world noise 或 joint/video 公共输出的变体不在本阶段适配。
- 基于 GWP 真实路径和该 FastWAM fixture 形成内部候选 PolicySpec；Gate B 前的 key、artifact 和 fixture 明确属于 development draft，不分配稳定 Proto/C ABI 契约。
- 实现该变体的开发版 converter、artifact loader、model factory、单一 `SessionImpl` 和必要的 architecture 私有步骤。
- 只补充两种模型确实共享的 CPU reference 函数，完成 image/state/text/action noise、normalized action 和最终 PolicyActionChunk 的分层 parity。
- 只实现真实 checkpoint 需要的能力；未验证的 world/video 输出和其他变体继续明确 unsupported。

### Review Gate B：第二架构审查

- 核对同一最小 PolicySpec 和公共函数能否覆盖 GWP-0.5/FastWAM，且 session 没有通过 `if environment` 选择计算路径。
- 核对公共函数没有 `if architecture`，environment adaptation 也没有判断模型名称。
- 核对公共 `policy_io` 函数确实被两个模型复用；仅一个模型使用的步骤退回对应 architecture 目录。
- 根据第二种架构暴露的问题调整内部 draft；审查通过后才冻结 0.5 PolicySpec、artifact schema、Proto 字段和主要公共接口。

### Phase 4：冻结 schema、工具和公共协议

- 冻结 PolicySpec GGUF key、enum、必填字段和字段间约束，并实现正式 reader/validator。
- converter 为 GWP 和已验证 FastWAM artifact 写入稳定 schema；loader 对 legacy 字段执行显式迁移或交叉校验。
- `ModelInfo` 暴露完整 PolicySpec，完成 Proto/C ABI 序列化、inspector 和 round-trip fixture。
- 按第 11 节实现并冻结 envelope/CanonicalObservation 的 Proto field number、WebSocket 状态机、连接级 session 生命周期和串行 predict 错误语义。
- 建立 converter、inspector、runtime 和 Proto 对同一 PolicySpec 的一致性测试；Gate B 前的 development artifact 不承诺兼容，必须重新转换。

### Phase 5：远程 eval 骨架和 RoboTwin 集成

- 在 `eval/sim/` 建立三个 ignored upstream checkout、六个 client/server 入口和三个固定 revision setup 入口；在 `eval/common/` 保存 WebSocket/Protobuf codec、连接状态机、共享 compatibility checker 与 model server core。
- 实现 RoboTwin client 内的 observation/action adaptation 和 payload validation；`run_robotwin_server.py` 提供 RoboTwin environment contract，由 server core 完成权威 compatibility check，继续使用远端 client/server 部署验证完整链路。
- server 返回完整 chunk；RoboTwin client 通过 `execute_steps=horizon` 保持当前执行语义，并验证 reset 分别清空 server session 与 client queue。
- LIBERO/LIBERO-X 入口在对应 profile 实现前保持显式未实现，不提供伪 action、默认转换或本地 runner fallback。

### Phase 6：逐步填充剩余 profile 和环境适配

- 完成 FastWAM 其余两个环境 profile，覆盖 2-view/7D 和 3-view/14D 两类契约。
- 填充 LIBERO、LIBERO-X client 的 observation/action adaptation、远端环境循环接入和 ActionChunkExecutor deployment config。
- GWP-0.5 LIBERO/LIBERO-X 权重完成后，先确认真实视觉 geometry，再转换 artifact、实现所需模型路径并完成 parity。
- 每增加一个 profile，只补充 converter metadata/stats、PolicySpec、必要公共函数或 architecture 私有步骤、compatibility check 和 fixture，不修改已经冻结的整体调用链。

### Phase 7：完整适配矩阵和发布验证

- 跑通 GWP-0.5/FastWAM 与三个环境的六个组合。
- checkpoint 尚在训练的组合可以先完成 schema、adapter 和合成 smoke test，但保持“未正式支持”状态。
- 对六个组合分别使用真实 checkpoint 完成数值对齐和 simulator rollout 后，才满足完整适配矩阵发布条件。
- 对所有真实 checkpoint 执行成功率和延迟评测。

### Phase 8：清理和发布

- 删除旧的模型/环境交叉 parser 和硬编码 shape。
- 在所有新 artifact 和调用方迁移完成后删除临时 legacy PolicySpec 构造路径；若决定长期兼容，则把它转成有版本测试的正式 migration layer。
- 更新 README、artifact 文档、启动脚本和 release checklist。
- 冻结 0.5 runtime/protocol/artifact/ABI 版本。

### 框架骨架验收条件

在开始批量填充 profile 前，至少满足：

1. 工程和现有测试可构建运行，新增目录没有反向依赖或循环所有权。
2. GWP-0.5 RoboTwin 已在单一 `SessionImpl::predict()` 内通过公共 policy 函数和 GWP 私有 engine 完成新链路，数值和性能无行为回归。
3. action shape 来自内部 PolicySpec/Prediction，不再由 RoboTwin server 硬编码 `[48,14]`。
4. registry 可以区分已实现和未实现 architecture/profile，未实现路径在显存大分配和推理前明确失败。
5. 公共层不存在模型/环境交叉分支，runtime/server 不持有 action execution queue。
6. 每个接口都有清楚的输入、输出、错误和 reset/lifecycle 测试，后续 profile 可以在不修改总体调用链的情况下增量接入。

## 16. 明确禁止的实现方式

- 在 `gwp05.cpp` 或 `fastwam.cpp` 中判断 `libero`、`liberox`、`robotwin`。
- 在环境 adaptation 代码中判断 `gwp05` 或 `fastwam`。
- 在 simulator client 和 environment server 中各维护一套互相可能漂移的 PolicySpec compatibility 规则。
- server 同时要求 GGUF 和独立 stats JSON，且二者可以不一致。
- 通过 artifact 文件名推断 view 数、normalization 或 action dim。
- 让公共 image ops 在 `predict()` 热路径读取 GGUF、解析 metadata，或根据 architecture/environment 名称选择 resize/crop。
- 为缺少相机的 checkpoint 静默补零图像。
- 对 normalization 缺失静默使用 identity。
- 将所有模型预处理做成一个包含大量 `if arch` 的 Python client。
- 把 gripper debounce、replan queue 或 simulator success 判断放进 architecture session。
- 在 C++ runtime/model server 中持有 action execution queue，或根据 `execute_steps` 截断 `Prediction.action`。
- 为了抽象而引入任意 preprocessing DSL 或过早的插件系统。
- 为所有模型强制定义统一 `PreparedInputs`/`CorePrediction`，或在真实复用出现前增加 `PolicyProcessor`、`ModelCoreSession`、`SimulatorRunner` 等基类。

## 17. 待继续确认的细节

### 17.1 本轮已冻结

- 0.5 最终目标仍是 GWP-0.5/FastWAM 对 LIBERO/LIBERO-X/RoboTwin 的六种组合；GWP-0.5 的 LIBERO/LIBERO-X 权重训练中，其他四种组合已有权重。
- PolicySpec 只描述 checkpoint 的外部输入输出与 pre/post-processing 契约，不保存 simulator、rollout 和模型内部 geometry。
- 环境 raw key/语义转换分别属于 observation/action adaptation；wire 传原始 instruction，serving request bridge 只使用当前 profile/部署已经验证的 language path；共有图像/state/action 数值处理由具体 session 调用 C++ 无状态 policy 函数；模型数学属于 architecture session 私有实现。
- 0.5 暂不设计 external-tokenizer manager 及其资源发现、下载、revision 解析、文件校验、容器挂载和生命周期。依赖该未决机制的 profile 显式 `Unsupported`，外部 tokenizer 部署不再作为 0.5 待确认后强行闭环的发布项。
- 0.5 保持 flat state API：adapter 按 `StateSpec.fields` 排序，C++ 只做 validate/pad/normalization。
- 图像预处理配置随 checkpoint 写入 GGUF，模型加载时一次性解析/校验为只读 `ImageSpec`；预测时具体 session 调用不依赖 `GgufReader` 的公共 policy image 函数执行。请求只提供命名的实际图像数据，环境 adapter 和 transport 不执行 checkpoint resize/composition/pixel-range/layout。per-view resize 使用语义明确的 `none`、`stretch` 或 `cover_center_crop`，图像组合使用 `none` 或通用 `canvas + placements`，不为环境布局增加专用分支。
- runtime/server 始终返回完整 `[horizon, real_action_dim]` PolicyActionChunk；最小 ActionChunkExecutor 位于 simulator side，`execute_steps` 不进入 PolicySpec，replan/ensemble 等尚未冻结的控制策略也不进入模型契约。
- checkpoint action 恢复属于完整 `SessionImpl::predict()` 的 policy postprocess，不属于 architecture 私有 graph、server 或环境 adapter。`ActionRecoverySpec` 支持 `identity` 和 `add_current_state`；后者仅使用 `reference_state_indices` 表达 raw-state 引用，`-1` 表示绝对维度，不再重复保存 `delta_mask`。当前 GWP-0.5 RoboTwin mapping 为 `[0,1,2,3,4,5,-1,7,8,9,10,11,12,-1]`，所有 horizon 行引用同一次请求 raw state。
- `Inputs.action_noise` 是可选的 F32 `[horizon, model_action_dim]` action denoise 初始噪声；`Capabilities.explicit_action_noise` 声明是否支持显式输入，缺省时由 session-owned RNG 生成并由 `SessionOptions.random_seed`/`session_reset()` 管理。
- FastWAM 0.5 只支持 action-only/action-noise 公共推理路径；video/world noise、video/world 输出及 joint/video 公共模式均为明确非目标，不能复用 `action_noise` 或通过 `Prediction.auxiliary` 半公开。
- 0.5 wire 固定为 WebSocket binary frame + Protobuf payload；首帧 Hello 完成 protocol/environment 握手，一条连接独占一个 session，同一 session 串行 predict，reset/close/断线均按连接管理 session 生命周期。
- environment/PolicySpec compatibility 的权威所有者是 server：三个 `run_*_server.py` 分别提供 environment id/contract，共享 server core 执行同一 checker。client 只负责 adaptation 和实际 payload validation；C++ runtime/model/policy 层仍不包含环境分支。
- `eval/sim/` 保存三个 revision-pinned upstream checkout 的忽略路径，以及每个环境的远程 client/server/setup 入口；`eval/common/` 保存 WebSocket/Protobuf codec、连接状态机、共享 compatibility checker 和 model server core。环境原始 observation/controller adaptation 仍位于 simulator client。
- 实施采用 framework-first：先建立内部骨架，用 GWP-0.5 RoboTwin 完成第一次无行为审查，再用 FastWAM 首个 profile 验证第二种架构，之后才冻结主要公共 schema 并批量填充其余 profile。

### 17.2 仍待确认

以下问题不阻塞 Phase 1 的内部框架骨架，但必须在对应 schema、profile、adapter 或发布阶段开始前逐项冻结：

1. GWP-0.5 LIBERO/LIBERO-X 权重完成后的视觉 slot、VAE/composition geometry，以及是原生两视图、动态视图还是其他结构。
2. 六个目标组合各自使用的 checkpoint 路径、revision、模型变体、训练数据和许可证。
3. `scene`、`wrist`、`left_wrist`、`right_wrist` 是否作为最终公共 image role 名称。
4. LIBERO 与 LIBERO-X action 的精确 frame、delta 定义和 gripper 正负约定。
5. RoboTwin 14 维 state/action 每一维的最终字段名称、单位和 controller contract；当前 GWP-0.5 的 mixed delta/absolute 恢复位置、顺序和 reference-state mapping 已冻结，不再属于待确认项。
6. 三套 FastWAM checkpoint 的准确 action horizon、video frame 数和 inference scheduler 参数，以及哪些 checkpoint/推理入口满足已冻结的 action-only/action-noise 公共契约。
7. normalization tensor 使用 real dim 还是 model/padded dim，以及 mask 的统一规则。
8. 各环境 `execute_steps` 的默认配置，以及 0.5 是否确实需要独立 gripper filter；replan/ensemble 暂不作为 0.5 公共组件。
9. protocol、artifact schema 和 C ABI 的最终版本号；wire transport/session 语义已冻结，不再属于候选项。
10. LIBERO-X 是否复用 LIBERO GGUF，还是必须使用独立微调 checkpoint 和统计量。
11. RoboTwin、LIBERO 和 LIBERO-X upstream checkout 的最终 URL、固定 revision、许可证和环境安装版本。

这些问题可以改变 profile 内容或环境转换行为，但不能改变本文确定的依赖方向：architecture session 只依赖模型结构和 PolicySpec；公共 policy 函数只依赖 PolicySpec 和 tensor/image 输入；observation/action adaptation 只依赖 simulator/controller 与 PolicySpec，不依赖模型名称；serving compatibility checker 只依赖 environment contract 与 PolicySpec，不进入 C++ model/policy 计算路径。
