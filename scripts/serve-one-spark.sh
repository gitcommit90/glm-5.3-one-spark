#!/usr/bin/env bash
set -euo pipefail
# patch_glm_video_placeholders installs a .pth import hook into the live site-packages,
# so it must run at container start. The other overlay patches are applied at image build.
python3 /opt/glm53/patch_glm_video_placeholders.py
K="${ONE_SPARK_K:-5}"  # DFlash2 draft depth; 5 = best prose/code, 8 = best structured (see README K sweep)
SPEC='{"method":"dflash","model":"/draft","num_speculative_tokens":'"$K"',"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":1}'
# ---- Runtime knobs (all default to the shipped recipe behaviour) ----
# ONE_SPARK_CTX / ONE_SPARK_UTIL / ONE_SPARK_SEQS / ONE_SPARK_MNBT: context, gpu-memory-utilization,
#   max-num-seqs, max-num-batched-tokens.
# ONE_SPARK_ASYNC=0|1: force --no-async-scheduling / --async-scheduling (unset = vLLM default, async for DFlash).
#   With async on, the DFlash2 drafter reserves a 2047+2*MNBT token in-flight window; each 64-token drafter
#   block is padded to a full 7168-token MLA page, so that window costs 257 blocks (13 GiB) for 262k context.
#   ONE_SPARK_ASYNC=0 halves it; bs=1 decode is unchanged.
# ONE_SPARK_APC=0|1: --no-enable-prefix-caching / --enable-prefix-caching (default 1).
# ONE_SPARK_DRAFT_BLOCK=N (e.g. 1024): raise the drafter's compact block in the padded slot-share path
#   (kv_cache_utils.py) so the drafter reserves ~10 blocks instead of 145/257. FlashAttention reports
#   MultipleOf(16) and select_common_block_size returns the manager block, so kernel block == manager block.
#   Measured lossless (acceptance 3.3-3.6, decode unchanged). Unset = shipped 64.
# ONE_SPARK_MAMBA_SEED_FIX=1 (default): fix for vllm-project/vllm#55600 — add_request seeds the mamba state
#   index with cache_config.block_size, which EngineCore lowers to the drafter's block (64/1024) while mamba
#   state lives in 7168-token blocks; prefix hits of >= 8 blocks then read past the block-table row (Xid 31)
#   and shorter hits silently restore the wrong KDA state. Fail-closed: if the anchor is missing the container
#   refuses to start (set ONE_SPARK_MAMBA_SEED_FIX=0 to start anyway; then also set ONE_SPARK_APC=0).
case "${ONE_SPARK_ASYNC:-}" in 0) ASYNC_FLAG=--no-async-scheduling ;; 1) ASYNC_FLAG=--async-scheduling ;; *) ASYNC_FLAG= ;; esac
case "${ONE_SPARK_APC:-1}" in 0) APC_FLAG=--no-enable-prefix-caching ;; *) APC_FLAG=--enable-prefix-caching ;; esac
if [ -n "${ONE_SPARK_DRAFT_BLOCK:-}" ]; then
  KVU=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
  sed -i "s/compact_block = 64$/compact_block = ${ONE_SPARK_DRAFT_BLOCK}/; s/s.block_size != 64 or s.page_size_padded != mla_page/s.block_size != ${ONE_SPARK_DRAFT_BLOCK} or s.page_size_padded != mla_page/" "$KVU"
  echo "[one-spark] drafter block patch: $(grep -c "compact_block = ${ONE_SPARK_DRAFT_BLOCK}" "$KVU") + $(grep -c "block_size != ${ONE_SPARK_DRAFT_BLOCK}" "$KVU") sites"
fi
if [ "${ONE_SPARK_MAMBA_SEED_FIX:-1}" = "1" ]; then
  MH=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  sed -i "s|(new_req_data.num_computed_tokens - 1) // self.cache_config.block_size|(new_req_data.num_computed_tokens - 1) // self.cache_config.mamba_block_size|" "$MH"
  N=$(grep -c "num_computed_tokens - 1) // self.cache_config.mamba_block_size" "$MH")
  echo "[one-spark] mamba seed fix (vllm#55600): $N site"
  if [ "$N" != "1" ]; then
    echo "[one-spark] FATAL: mamba seed fix not applied (expected 1 match, got $N) - did the image change? Without it a prefix-cache hit of >= 8 blocks faults (Xid 31). Set ONE_SPARK_MAMBA_SEED_FIX=0 ONE_SPARK_APC=0 to start deliberately." >&2
    exit 97
  fi
fi
exec vllm serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-2.05 \
  --host "${ONE_SPARK_HOST:-127.0.0.1}" --port "${ONE_SPARK_PORT:-18080}" \
  --tensor-parallel-size 1 \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  $APC_FLAG --no-enable-flashinfer-autotune \
  --quantization exl3 \
  --max-model-len "${ONE_SPARK_CTX:-262144}" \
  --gpu-memory-utilization "${ONE_SPARK_UTIL:-0.90}" \
  --max-num-seqs "${ONE_SPARK_SEQS:-4}" --max-num-batched-tokens "${ONE_SPARK_MNBT:-7168}" \
  $ASYNC_FLAG \
  --kv-cache-dtype fp8 \
  --speculative-config "$SPEC" \
  --chat-template /opt/glm53/chat_template.jinja \
  --limit-mm-per-prompt '{"image":2,"video":0}' --skip-mm-profiling \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32
