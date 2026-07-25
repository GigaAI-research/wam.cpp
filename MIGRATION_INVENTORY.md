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
release/patches/llama-b9866-native-bf16.patch
release/patches/llama-b9866-p7-fusions.patch
release/src/models/common/gguf_reader.*
donor/scripts/common/gwp05_gguf.py
donor/scripts/convert/convert_gwp05.py
donor/scripts/inspect/inspect_gguf.py
donor/tests/integration/phase3_artifacts.cpp
donor/tests/unit/test_converter.py
```

依赖/patched GGUF runtime 可以从 release 恢复 provenance；reader、converter 和 inspector 必须适配 0.5 PolicySpec，不恢复旧 metadata 作为新 schema。

新增目标文件：

```text
src/policy/policy_spec_loader.cpp
src/policy/policy_spec_loader.h
```

Slice 2 不迁入 GWP engine 或 serving。

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

## 8. Slice 4：GWP 私有 engine

以下文件作为一个 `IMPORT_PRIVATE` 单元：

```text
src/models/gwp05/engine/engine.cpp
src/models/gwp05/engine/engine.h
src/models/gwp05/engine/weights.cpp
src/models/gwp05/engine/text_encoder.cpp
src/models/gwp05/engine/vae.cpp
src/models/gwp05/engine/mot.cpp
src/models/gwp05/engine/cache.cpp
src/models/gwp05/kernels/cuda_preprocess.cu
src/models/gwp05/kernels/cuda_preprocess.h
```

新增私有共享头：

```text
src/models/gwp05/engine/engine_internal.h
```

迁入时允许机械移动和私有接口整理，不允许改变数学。旧 engine 使用的 `.cpp` include 必须转换成正常编译单元。所有变更用 donor 中间 tensor/replay 做 parity。

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

Slice 7 按 0.5 contract 新建 Proto、codec、server core 和测试。C ABI 是否同时恢复由当时真实调用方决定，不作为新 wire 的前置条件。

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

完成 Slice 0 cleanup commit 后，只开始 Slice 1：设计正式 `wam_core`、最小测试 target、registry unsupported path 和 lifecycle contract。GGUF dependency 和任何 GWP engine 文件留到后续 Slice。
