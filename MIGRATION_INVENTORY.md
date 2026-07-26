# wam.cpp 0.5 代码迁入盘点

## 1. 目的

本文档记录 release、donor 和 0.5 skeleton 的文件差异，并把旧文件分配到 skeleton-first 的 Slice 0-8。它回答“何时参考或迁入哪些文件”，不授权全树复制，也不替代 `ARCHITECTURE.md` 的模块职责。

比较对象：

| 名称 | 路径或 revision | 角色 |
| --- | --- | --- |
| release | `public-upstream/main@7adca01`，本地 `../wam.cpp-release/` | Git provenance 和公开 release 参考 |
| donor | `/testessfs10/users/yejun.zeng/codes/gwp/wam.cpp/` | 可运行 GWP engine、测试和数值 reference |
| target | `private-origin/refactor/0.5`，即当前目录 | 唯一开发目录 |

比较排除 `.git/`、`__pycache__/`、`*.pyc` 和空目录。

## 2. 差异结论

### 2.1 Release 与 donor

| 指标 | 结果 |
| --- | ---: |
| release tracked files | 77 |
| donor effective files | 204 |
| 内容完全相同 | 63 |
| donor 同路径有差异 | 14 |
| donor 新增 | 127 |
| release 独有 | 0 |

GWP engine 的唯一同路径差异是 `cache.cpp` 末尾空行；engine 数学、权重、graph、scheduler 和 kernel 没有内容差异。donor 的其他同路径差异集中在 CMake 测试开关、CI、发布脚本、README 和 LICENSE。

因此：

- runtime 文件有 release Git 来源时，保留该 provenance；
- donor 作为可运行数值 reference 和新增测试来源；
- 不存在需要“整树恢复 donor 才能得到更新 engine”的理由。

### 2.2 清理前 target

原 worktree 有 110 个 tracked files：56 个 skeleton 文件和 54 个隐式 release 文件。Slice 0 保留其中 Apache-2.0 `LICENSE`，删除其余 53 个旧文件，并加入本清单。

## 3. 迁入状态

| 状态 | 含义 |
| --- | --- |
| `KEEP` | skeleton/0.5 权威文件，禁止旧代码覆盖 |
| `REFERENCE` | 只用于阅读、diff 或 parity，不直接复制 |
| `ADAPT` | 按 0.5 边界迁入相关逻辑，不能整文件盲拷贝 |
| `IMPORT_PRIVATE` | 作为 GWP architecture 私有数学整体迁入 |
| `LOCAL_ONLY` | 可用于本地验证，不提交 Git |
| `DEFER` | 后续 Slice 再决定 |
| `DROP` | 旧架构、生成物或空目录，不迁入 |

## 4. Slice 0 保留清单

### 4.1 Skeleton 文件

以下 56 个原始文件保持 `KEEP`：

```text
.gitignore
ARCHITECTURE.md
CMakeLists.txt
README.md
plan.md

include/wam/types.h
include/wam/version.h
include/wam/wam.h

src/arch.cpp
src/arch.h
src/model.cpp
src/model_internal.cpp
src/model_internal.h
src/model_registry.cpp
src/model_registry.h

src/policy/policy_spec.cpp
src/policy/policy_spec.h
src/policy/image_ops.cpp
src/policy/image_ops.h
src/policy/state_ops.cpp
src/policy/state_ops.h
src/policy/action_ops.cpp
src/policy/action_ops.h

src/models/common/gguf_reader.cpp
src/models/common/gguf_reader.h

src/models/gwp05/artifact.cpp
src/models/gwp05/artifact.h
src/models/gwp05/inputs.cpp
src/models/gwp05/inputs.h
src/models/gwp05/semantics.cpp
src/models/gwp05/semantics.h
src/models/gwp05/model.cpp
src/models/gwp05/model.h
src/models/gwp05/engine/engine.cpp
src/models/gwp05/engine/engine.h

src/models/fastwam/artifact.cpp
src/models/fastwam/artifact.h
src/models/fastwam/inputs.cpp
src/models/fastwam/inputs.h
src/models/fastwam/semantics.cpp
src/models/fastwam/semantics.h
src/models/fastwam/model.cpp
src/models/fastwam/model.h
src/models/fastwam/engine/engine.cpp
src/models/fastwam/engine/engine.h

eval/common/rpc.py
eval/common/server.py
eval/sim/run_robotwin_client.py
eval/sim/run_robotwin_server.py
eval/sim/run_libero_client.py
eval/sim/run_libero_server.py
eval/sim/run_liberox_client.py
eval/sim/run_liberox_server.py
eval/sim/setup_robotwin.sh
eval/sim/setup_libero.sh
eval/sim/setup_liberox.sh
```

额外保留：

```text
LICENSE                    # Apache-2.0，与 public release 一致
MIGRATION_INVENTORY.md
```

### 4.2 清理掉的隐式 release 内容

以下内容已从 skeleton 起点删除，但可从 Git 历史恢复：

```text
.github/
apps/
cmake/
integrations/
patches/
proto/
requirements/
scripts/
src/serving/
CONTRIBUTING.md
include/wam/c_api.h
src/models/common/input_validation.*
src/models/common/scheduler.*
src/models/gwp05/engine/{cache,mot,text_encoder,vae,weights}.cpp
src/models/gwp05/kernels/
```

删除这些文件不表示永不使用其逻辑，只表示它们必须在所属 Slice 中显式迁入，不能作为隐藏旧框架留在树中。

## 5. Slice 1：构建和测试框架

### 5.1 参考文件

```text
release/CMakeLists.txt
release/cmake/Dependencies.cmake
release/cmake/WamModels.cmake
release/cmake/WamOptions.cmake
donor/tests/CMakeLists.txt
donor/tests/contracts/
donor/tests/integration/model_extensibility.cpp
donor/tests/integration/runtime_core.cpp
donor/tests/support/
```

全部状态为 `REFERENCE/ADAPT`。不得恢复旧 source list、apps、serving、Proto、install 逻辑或 v0.1 强制 GWP 假设。

Slice 1 新建/调整：

- 0.5 `wam_core` source list；
- `WAM_BUILD_GWP05/FASTWAM/TESTS`；
- 最小 `tests/contracts`、`tests/unit`、`tests/support`；
- registry unsupported 和 lifecycle contract。

不迁入：real replay、converter、CUDA、C ABI、Proto、server。

## 6. Slice 2：GGUF 和 PolicySpec

### 6.1 参考/适配文件

```text
release/cmake/ApplyLlamaPatch.cmake
release/cmake/Dependencies.cmake
release/patches/llama-b9866-native-bf16.patch  # Slice 4 reference only
release/patches/llama-b9866-p7-fusions.patch   # Slice 4 reference only
release/src/models/common/gguf_reader.*
donor/scripts/common/gwp05_gguf.py
donor/scripts/convert/convert_gwp05.py
donor/scripts/inspect/inspect_gguf.py
donor/tests/integration/phase3_artifacts.cpp
donor/tests/unit/test_converter.py
```

Slice 2 只复用 release pinned llama.cpp `b9866` revision/archive hash，不应用与模型计算有关的 BF16/fusion patch，也不构建 CPU/CUDA backend。reader 适配 0.5 internal PolicySpecDraft，不恢复旧 metadata 作为新 schema；patch 在 Slice 4 engine 迁入时按数值 reference 单独评估。

loader/validator 保持在已有的：

```text
src/policy/policy_spec.cpp
src/policy/policy_spec.h
```

Slice 2 增加结构化 test-only GGUF fixture builder，不提交生成的二进制 artifact。converter/inspector 和六个真实 profile fixture 在对应 architecture contract/schema freeze 阶段实现。Slice 2 不迁入 GWP engine 或 serving。

## 7. Slice 3：GWP contract

以下 donor/release 文件状态为 `ADAPT`：

```text
src/models/gwp05/artifact.*
src/models/gwp05/semantics.*
src/models/gwp05/inputs.*
src/models/common/input_validation.*
src/models/common/scheduler.*
tests/unit/gwp05_semantics.cpp
tests/unit/kernel_dispatch.cpp
```

迁入职责限于 artifact metadata、GWP geometry、input validation 和 scheduler contract。不得在 Slice 3 带入 weights、graph、CUDA 或旧 public types。

Slice 3 已完成上述 `ADAPT` 范围：新 draft PolicySpec 与已审计 32D quantile dual-arm legacy migration 均可 metadata-only load；legacy artifact 缺少可靠 dataset provenance，因此不再标成 RoboTwin。GWP 私有 geometry/旧重复字段交叉校验，input tensor contract、prompt/MoT structural semantics 和 flow-match scheduler 已由合成 fixture 覆盖。没有迁入 donor engine、weights、graph、kernel、旧 API 或真实 replay payload。`tests/unit/kernel_dispatch.cpp` 依赖 engine dispatch，保留到 Slice 4，不在 Slice 3 用空实现占位。

Slice 4A 增加 `LOCAL_ONLY` 真实 Gate：四个 22–34 GiB GGUF 的 legacy F32、MoT BF16、MoT+VAE BF16 和 packed-QKV descriptor/policy、legacy PolicySpec 与 metadata-only lifecycle 已通过；donor complete-MoT F32 CPU reference 与独立 CUDA prefix-cache reference 由 path-free manifest 固定。Gate A 又为当前 14D z-score RoboTwin checkpoint 生成正式 schema-v2 PolicySpec packed-BF16 GGUF，并通过独立 inspector、artifact loader、metadata lifecycle 与 A800 public CUDA/BF16 Session 数值 parity。该结论来自独立 14D fixture，未由 legacy Gate 外推。

## 8. Slice 4：GWP 私有 engine

以下 donor/release 文件作为一个数值上不可拆散的 `IMPORT_PRIVATE` 来源单元：

```text
donor/src/models/gwp05/engine/engine.cpp
donor/src/models/gwp05/engine/engine.h
donor/src/models/gwp05/engine/weights.cpp
donor/src/models/gwp05/engine/text_encoder.cpp
donor/src/models/gwp05/engine/vae.cpp
donor/src/models/gwp05/engine/mot.cpp
donor/src/models/gwp05/engine/cache.cpp
donor/src/models/gwp05/kernels/cuda_preprocess.cu
donor/src/models/gwp05/kernels/cuda_preprocess.h
```

target 不沿用 donor 的组件文件名作为顶层结构，而统一成 architecture engine 职责：

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
    cuda_preprocess.cu
    cuda_preprocess.h
```

具体映射为：`text_encoder.cpp -> language_encoder.cpp`；`vae.cpp -> observation_encoder.cpp`；`mot.cpp` 的 joint block 与 action/denoise 分别进入 `backbone.cpp` 和 `action.cpp`；`weights.cpp` 的 tensor binding 保留在 `weights.cpp`，backend/graph/buffer 部分进入 `runtime.cpp`；`cache.cpp` 继续承担可选 prefix/KV/graph cache。T5、VAE、MoT 和具体 kernel 名称保留在类型、函数、权重前缀和 reference stage 中。

每个 architecture 的 facade 统一为 model-level `create_engine()`、轻量 `create_engine_session()`、`predict()` 和 `reset()`，但 engine/session 具体类型保持 architecture-private，不建立跨模型 `LanguageEncoder`/`Backbone`/`ActionHead` 基类，也不共享 prepared/core tensor 类型。迁入时允许机械移动和私有接口整理，不允许改变数学。旧 engine 使用的 `.cpp` include 必须转换成正常编译单元。所有变更用 donor 中间 tensor/replay 做 language、observation、backbone、action 和 cache 分阶段 parity。

Slice 4B 已完成上述 `IMPORT_PRIVATE` 的 target 重组：所有 donor `.cpp` include 已删除，权重绑定、backend/runtime、T5、VAE、MoT、action denoise 和 prefix cache 分别进入声明的职责文件；pinned fusion/native-BF16 patch 与当时的 CUDA preprocess 已进入 opt-in CUDA build。legacy F32 external-embedding complete-MoT 通过真实 GGUF 的逐阶段 donor parity，engine 返回 normalized `[48,32]` action，并验证 reset 和显式 noise 确定性。Slice 6 为建立统一 policy boundary 已删除跨越 raw-image resize/canvas 与 GWP patchify 的旧 fused preprocess；CUDA engine 现从公共 CPU reference composite 上传，未来设备优化必须只实现等价公共 image ops 或 architecture-private patchify。CUDA + cuDNN native-BF16 继续要求完整编译和链接，但当前容器没有可用 GPU，CUDA prefix-cache、BF16 和 token/fixed prompt 的独立数值 Gate 仍保持明确未验证，不能由编译成功替代。

测试所有权固定为：artifact test 保持 metadata-only；weights/runtime test 负责 backend 和完整 binding；engine parity 到 normalized/model-space action 为止；公共 action decode 和 Session/serving 分别属于 Slice 5–7。engine 的 vision/text/prefill/decode timing 按计算阶段统计，T5/VAE/MoT/cache/denoise step 的真实名称进入 `model_timings`。

不迁入：apps、C API、Proto、server、environment integration。

## 9. Slice 5：GWP Model/Session

以下文件状态为 `ADAPT`：

```text
src/models/gwp05/model.*
src/model.cpp
src/model_internal.*
src/model_registry.*
tests/integration/runtime_core.cpp
tests/integration/model_probe.cpp
tests/parity/gwp05_replay.cpp
```

旧代码只提供 load/session/cache/error 的行为参考。0.5 public types、registry 和 `SessionImpl::predict()` 是权威边界，不能恢复 `active_session`、旧 noise 字段或旧 ModelInfo。

Slice 5 已完成该 `ADAPT`：model-level private engine 唯一拥有 backend/weights，轻量 engine session 独立拥有 graph/cache/workspace，architecture `Gwp05SessionImpl` 拥有请求 RNG，公共 Model/Session 已接通 load/create/predict/reset/free。共享 backend execution 串行化，但没有 session 切换式全局 cache；model handle 提前释放时资源由现存 session 延长生命周期。Slice 5 的 legacy F32 私有 action adapter 已在 Slice 6 被通用 policy ops 替换。

## 10. Slice 6：Policy boundary

target 中以下文件保持 `KEEP` 并填充实现：

```text
src/policy/image_ops.*
src/policy/state_ops.*
src/policy/action_ops.*
```

donor GWP engine 中对应逻辑只作为 `REFERENCE`：

- image resize/crop/composition；
- state quantile normalization/padding；
- action noise fallback；
- action trim/unnormalize/raw-state offset。

迁移完成后从 engine 删除这些职责。不得在 policy ops 中判断 `gwp05`、`fastwam` 或 environment name。

Slice 6 已完成该边界迁移：

- `image_ops` 校验/排序 named RGB view，执行 `none/stretch/cover_center_crop`、nearest/bilinear/bicubic reference、antialias、canvas、pixel range 和 CHW/HWC layout；legacy GWP migration 已从错误的 `stretch` 修正为 donor 实际使用的 `cover_center_crop`。
- `state_ops` 保留 raw state，并生成按 mask 处理的 padded/normalized model state；`action_ops` 执行 model action 校验、`trim -> clip -> unnormalize -> raw-state recovery` 和最终 tensor 构造。
- action noise 由无状态公共函数校验或生成，RNG 本体与初始 seed 由具体 session 持有；显式 noise 不推进 RNG，reset 恢复 seed。
- GWP engine 只接收 RGB CHW `[-1,1]` composite、model-space state 和完整 action noise，只返回 normalized/model-space action；VAE latent normalization、2x2 patchify、T5、MoT、scheduler 和 cache 仍为 architecture 私有。
- 公共 policy 层没有 model/environment 分支；私有 artifact contract 不再复制 state/action normalization statistics。

## 11. Slice 7：新 serving

以下旧文件状态为 `REFERENCE`，不直接恢复：

```text
release/proto/wam.proto
release/include/wam/c_api.h
release/src/serving/
release/scripts/serving/
release/integrations/giga-world-policy/
donor/eval/robotwin/inference_server_wam.py
```

原因：旧链路是旧 Proto/C ABI/gRPC 或 GWP-specific server，不满足 WebSocket binary + Protobuf、Hello-first、连接级 session 和 environment-specific server contract。

Slice 7 按 0.5 contract 新建 Proto、structured C ABI、codec、server core 和测试；没有恢复旧 C ABI method dispatch、gRPC service 或 msgpack wire。

## 12. Slice 8：FastWAM 和 eval

### 12.1 FastWAM

donor 中 `src/models/fastwam/` 只有空目录，没有可迁入实现。target skeleton 是唯一入口，真实实现必须来自 FastWAM checkpoint/reference 工程的单独盘点。

FastWAM 首个 profile 验证 PolicySpec/policy ops 后执行 Gate B，再冻结稳定 schema。

### 12.2 Eval

target 的以下文件保持 `KEEP`：

```text
eval/common/rpc.py
eval/common/server.py
eval/sim/run_*_client.py
eval/sim/run_*_server.py
eval/sim/setup_*.sh
```

donor 的旧 RoboTwin server 仅用于 observation/RPC 行为对照，不能成为共享 server core。

## 13. Tests 和 replay 归属

### 13.1 可适配迁入的测试源码

```text
tests/contracts/
tests/install-consumer/
tests/integration/
tests/kernels/
tests/parity/
tests/performance/
tests/support/
tests/unit/
tests/data/gwp05/README.md
tests/data/gwp05/replay.proto
```

测试按对应 Slice 逐个迁入，不在 Slice 1 全量复制。引用旧 public API/Proto 的测试必须重写为 0.5 contract。

### 13.2 LOCAL_ONLY

```text
/testessfs10/users/yejun.zeng/codes/gwp/wam.cpp/tests/data/gwp05/fixed/
/testessfs10/users/yejun.zeng/codes/gwp/wam.cpp/tests/data/gwp05/independent/
```

这两个目录约 6.5 MB，redistribution approval 尚未记录。它们不能提交到个人私有 GitHub；测试通过显式外部 root 读取，未提供时相关测试 skip/not registered。

以下原始报告同样不迁入：

```text
donor/docs/performance/v0.1/raw/
donor/reports/rtx4090/
```

## 14. 仓库级文件

### 14.1 `.gitignore`

保持 target 合并后的规则：build、binary、Python cache、模型、GGUF、profile、内部 replay 和 simulator checkout 均忽略；测试源码不整体忽略。

### 14.2 `README.md`

保持 Phase 1 skeleton README，随实际 Slice 进度更新。不得恢复 donor v0.1 README。

### 14.3 `LICENSE`

保持 public release 的 Apache-2.0。donor 根 LICENSE 是 MIT，不能覆盖。最终许可证和第三方 NOTICE 在公开发布前单独审查。

### 14.4 CI 和发布脚本

`.github/`、`scripts/release/` 和 `CONTRIBUTING.md` 在基础构建稳定后重新设计。旧文件只作参考，不能提前让 CI 固化尚未完成的 0.5 命令。

## 15. DROP/DEFER

当前不迁入：

```text
donor/docs/plan*.md
donor/docs/development/phase*-review.json
donor/docs/performance/v0.1/raw/
donor/reports/
donor/ROADMAP.md
donor/third_party/                  # 空目录
donor/src/kernels/                  # 空目录
donor/scripts/{benchmark,calibrate,quantize}/  # 空目录
donor/examples/                     # 空目录
__pycache__/
*.pyc
```

有价值的历史事实经验证后归纳进 0.5 文档，不同时维护多套权威计划。

## 16. 提交顺序

```text
Slice 0: Clean the 0.5 skeleton baseline
Slice 1: Establish the core build and contract tests
Slice 2: Implement GGUF-backed PolicySpec loading
Slice 3: Implement the GWP05 artifact contract
Slice 4: Import the private GWP05 engine
Slice 5: Connect the GWP05 model and session
Slice 6: Move GWP05 policy boundaries
Slice 7: Add the 0.5 WebSocket Protobuf server
Slice 8: Add FastWAM and simulator evaluation
```

每个 Slice 可以包含多个小提交，但不得跨 Slice 偷带旧文件。

## 17. 执行约束

1. 禁止全树 rsync 和 `rsync --delete`。
2. 每个迁入文件必须能映射到一个 Slice 和 owner。
3. runtime 数学有 release Git 来源时保留 provenance；donor 用于 parity。
4. 内部 replay、checkpoint、GGUF 和输出二进制不进入 Git。
5. 未实现路径 fail-fast，不使用 placeholder action。
6. public framework 不恢复旧 API 作为长期 compatibility layer。
7. engine 私有临时 adapter 必须在 Slice 6 删除。
8. FastWAM/GWP 和 simulator 不互相判断名称。

## 18. 下一步

Slice 6、Gate A 和 Slice 7 已完成。legacy F32 gate、CPU/CUDA build matrix、14D z-score RoboTwin 正式 GGUF、独立 PyTorch BF16 oracle、public Session parity、C ABI 和 WebSocket/Protobuf 状态机均有证据；不恢复 release gRPC/msgpack server。正式 token path RPC parity 通过，`beat_block_hammer` 跨容器单 episode smoke test 为 1/1，固定 manifest 100 次正式评测为 `87/100`，且没有 RPC/runtime failure。下一步进入 FastWAM Gate B，并复用相同 manifest 和延迟口径验证公共抽象。
