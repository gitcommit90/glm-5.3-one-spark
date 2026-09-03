# Startup time: 14 min → 3 min 21 s (2026-09-03)

Cold start to API-ready on one DGX Spark, same General23 image, same throughput.

| Build | Unloaded → `/health` 200 | Weight load | Notes |
|---|---:|---:|---|
| General23 baseline | ~14 min (profiled run: 1233 s) | 705 s | |
| L2 | 361 s | 60 s + 15 s draft | DNS fix + loader fix |
| **L3** | **201 s** | 74 s cold / ~60 s warm | + stop timeout, lean start script, skip graph-memory estimate |

Raw logs and the py-spy profile are in [`benchmarks/raw/startup-time-20260903`](../benchmarks/raw/startup-time-20260903).

## Where the 14 minutes went (py-spy, 50 Hz, whole startup)

| Cause | Time | Fix |
|---|---:|---|
| `torch.distributed.new_group(gloo)` blocked in `getaddrinfo` — container hostname lookup took 10 s (IPv6/tailscale search domains), 2 lookups × ~15 groups | 5 min | `--add-host $(hostname):127.0.1.1` and `-e GLOO_SOCKET_IFNAME=lo` |
| `exl3.py` weight loader: 148,608 expert tensors each copied straight from the safetensors mmap into a slice of the GPU parameter. Pageable `cudaMemcpy` from a not-yet-faulted mmap costs ~3.4 ms per tensor | 8.5 min | `clone()` the tensor into anonymous RAM before `dest.copy_()` — 0.35 ms per tensor. One line in `overlay/exl3.py` |
| Old container ignored SIGTERM and ran out the 90 s `--stop-timeout` | 92 s | `--stop-timeout 10` |
| Six overlay patch scripts re-run at container start; all were already applied at image build and only printed "already present" — each imports vLLM (~6 s) | 47 s | Removed from `scripts/serve-one-spark.sh` |
| CUDA-graph memory estimation profile pass | 48 s | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. Side effect: KV pool grows 20.75 → 23.55 GiB because the estimator over-reserved |

Disk was never the limit: NVMe reads at 5 GB/s and a full 79 GB safetensors pass takes 17 s.

## Tried and rejected

- `--safetensors-load-strategy eager`: reads whole shards into RAM. On unified memory that RAM *is* the GPU's memory; available KV fell to 6.9 GiB and the engine refused to start. Also slower (122 s). Keep the default mmap path.

## Verification after L3

- 17×23, prose, strict JSON all correct
- Structured C1, five runs, temperature 1.0: median 62.0 tok/s, accept ratio 0.94–0.98 (baseline 62.6 under the same sampling)
- KV pool 23.55 GiB / 368,268 tokens

## Remaining budget (~200 s)

27 s process spawn + imports · ~60–74 s weight load · 18 s memory profile · 33 s CUDA-graph capture · ~17 s API finalize · misc. The weight load could reach ~25 s with `safe_open(device="cuda")` in the loader; not done yet.
