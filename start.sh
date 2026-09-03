#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/GLM-5.3-Flash-exl3-2.05bpw}"
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/GLM-5.3-Flash-DFlash2}"
IMAGE="${IMAGE:-glm53-one-spark-vllm:tp1-general23}"
CONTAINER="${CONTAINER:-glm53-one-spark}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"

[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  echo "Target checkpoint missing at $MODEL_DIR; run ./download.sh first." >&2; exit 2;
}
[[ -f "$DFLASH_DIR/config.json" ]] || {
  echo "DFlash2 checkpoint missing at $DFLASH_DIR; run ./download.sh first." >&2; exit 2;
}
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 2; }
docker info >/dev/null

if [[ "${BUILD:-0}" == 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build --progress=plain \
    --build-arg GLM53_RECIPE_STAMP=tp1-general23-exl3-mul1-fused \
    -t "$IMAGE" "$ROOT"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --gpus all --network host --ipc=host \
  --shm-size 32g --stop-timeout 10 --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --add-host "$(hostname):127.0.1.1" \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a \
  -e EXL3_FUSED_MOE=1 \
  -e EXL3_MOE_ROW_TILE=0 \
  -e EXL3_TEMP_ROWS_FUSED=128 \
  -e EXL3_FAT_SORTED=0 \
  -e EXL3_FAT_BATCHED=0 \
  -e EXL3_FAT_KERNEL=1 \
  -e GLM53_SUPPRESS_STOPS_IN_REASONING=1 \
  -e GLM53_MIXED_PREFILL_CHUNK=skip \
  -e GLM53_INDEXER_WORKSPACE=stock \
  -e GLM53_SPINWAIT_MS=stock \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e ONE_SPARK_HOST="$HOST" -e ONE_SPARK_PORT="$PORT" \
  -v "$MODEL_DIR:/model:ro" -v "$DFLASH_DIR:/draft:ro" \
  -v "$ROOT/scripts/serve-one-spark.sh:/start.sh:ro" \
  -v "${CACHE_ROOT:-$HOME/.cache/glm53-one-spark}/vllm:/root/.cache/vllm" \
  -v "${CACHE_ROOT:-$HOME/.cache/glm53-one-spark}/triton:/root/.triton/cache" \
  -v "${CACHE_ROOT:-$HOME/.cache/glm53-one-spark}/tilelang:/root/.tilelang/cache" \
  -v "${CACHE_ROOT:-$HOME/.cache/glm53-one-spark}/torchinductor:/tmp/torchinductor_root" \
  --entrypoint bash "$IMAGE" /start.sh

echo "Container started. Initial API readiness takes about 3.5 minutes (was 14 before 2026-09-03)."
echo "Follow startup: docker logs -f $CONTAINER"
