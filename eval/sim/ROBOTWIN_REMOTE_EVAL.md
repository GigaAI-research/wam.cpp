# RoboTwin remote evaluation

The 0.5 evaluator uses one binary WebSocket message per Protobuf envelope. The
model server owns one C++ session per connection; the simulator client owns the
episode, observation adapter, action queue, and controller calls.

Build the CUDA runtime and descriptor inside `gwp_zyj`:

```bash
docker exec -it gwp_zyj bash
cd /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5
cmake -S . -B build-serving -G Ninja \
  -DWAM_CUDA=ON -DWAM_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DWAM_BUILD_SERVING=ON
cmake --build build-serving -j4
```

Start the server on an unused GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python eval/sim/run_robotwin_server.py \
  --library build-serving/libwam_c_api.so \
  --descriptor build-serving/wam.desc \
  --model /testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf \
  --tokenizer /testessfs10/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/tokenizer \
  --backend cuda --precision bf16 --host 0.0.0.0 --port 18060
```

Resolve the `gwp_zyj` container address from the host:

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' gwp_zyj
```

Then run one `beat_block_hammer` episode in `robotwin_zyj`, replacing
`SERVER_HOST` with that address:

```bash
docker exec -it robotwin_zyj bash
/root/miniconda3/bin/conda run -n RoboTwin python \
  /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/eval/sim/run_robotwin_client.py \
  --robotwin-root /testessfs10/users/yejun.zeng/codes/RoboTwin \
  --descriptor /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/build-serving/wam.desc \
  --task beat_block_hammer --episodes 1 \
  --host SERVER_HOST --port 18060 --execute-steps 48
```

The local tokenizer path is explicit and server-owned. No tokenizer is
downloaded or selected from untrusted client metadata. Tokenization runs in
Python; the artifact-resident T5 and all model inference remain in C++.

The initial 2026-07-25 smoke test used seed `100000`. The policy reached
RoboTwin success at environment step 106 (`1/1`). This confirms the remote
integration path only.

## Frozen 100-episode benchmark

The formal `beat_block_hammer` run uses the existing cross-model RoboTwin
manifest at:

```text
logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/manifest/eval_manifest_seed0_test100_unseen.json
```

Its SHA256 must be:

```text
5e104b97e6a920bc2cc6fbc28aee278f50a5c381e611d46ed22af1968b6b7c57
```

Do not let the evaluator generate or merge a new manifest when comparing model
success rates. With the server running, execute:

```bash
/root/miniconda3/bin/conda run --no-capture-output -n RoboTwin python -u \
  /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/eval/sim/run_robotwin_client.py \
  --robotwin-root /testessfs10/users/yejun.zeng/codes/RoboTwin \
  --descriptor /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/build-serving/wam.desc \
  --task beat_block_hammer --episodes 100 \
  --host SERVER_HOST --port 18060 --execute-steps 48 \
  --metrics-output /testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/metrics/wam05_seed0_test100.json
```

The 2026-07-25 run used CUDA BF16 on one NVIDIA A800 80GB PCIe, with the
RoboTwin client and wam.cpp server in separate containers. It completed all 100
episodes without an RPC/runtime failure:

| metric | result |
| --- | ---: |
| success | `87/100` (`87.0%`) |
| action-chunk requests | `378` |
| peak device memory | `23,512,685,184` bytes (`21.90 GiB`) |

Request latency is measured around `predict`, excluding simulator stepping.
`client infer` additionally includes observation image conversion. Reset makes
the first request rebuild the text/session cache; the second request completes
CUDA graph setup; request three and later are steady state.

| request group | samples | client infer mean / p50 / p95 (ms) | RPC mean / p50 / p95 (ms) | server total mean / p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| all | 378 | 185.19 / 203.97 / 283.84 | 183.64 / 202.47 / 282.39 | 177.85 / 197.34 / 276.91 |
| first after reset | 100 | 221.43 / 207.15 / 258.85 | 219.81 / 205.51 / 257.34 | 214.17 / 199.89 / 252.08 |
| second / graph setup | 100 | 243.13 / 215.89 / 291.38 | 241.59 / 214.37 / 289.78 | 235.65 / 208.71 / 282.68 |
| steady state | 178 | 132.29 / 132.05 / 136.02 | 130.77 / 130.50 / 134.56 | 124.98 / 124.99 / 127.46 |

Failed seeds were `100003`, `100012`, `100034`, `100055`, `100060`,
`100063`, `100081`, `100098`, `100100`, `100105`, `100114`, `100126`, and
`100131`. The full result is stored in `_result.txt`; the metrics JSON contains
the manifest digest, failure episode numbers, all request timings, grouped
distributions, and peak memory. The adjacent `.requests.jsonl` is written after
each request so a long benchmark retains partial timing evidence if interrupted.

The artifacts from this run are:

```text
/testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/20260725_153523/_result.txt
/testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/20260725_153523/failed_seeds.txt
/testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/metrics/wam05_seed0_test100.json
/testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/wam05/demo_clean/none/metrics/wam05_seed0_test100.requests.jsonl
```
