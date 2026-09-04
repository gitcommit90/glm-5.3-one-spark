#!/bin/bash
# Phase 1a: structured K sweep. Baseline K=7 on the running server first (no restart), then K=4 5 6 8, then restore K=7.
R=/home/tux/glm53-one-spark-runtime; S=$R/sweep-k; OUT=$S/results-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; ln -sfn $OUT $S/results.latest
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/sweep.log; }
bench(){ local k=$1 tag=$2; log "bench K=$k ($tag)"; BENCH_K=$k python3 $S/bench_k.py --phase structured-k$k --structured --runs 3 --max-tokens 400 --out $OUT/k$k-$tag.json > $OUT/k$k-$tag.stdout 2>&1; python3 - <<PY | tee -a $OUT/sweep.log
import json;d=json.load(open("$OUT/k$k-$tag.json"));print("RESULT K=$k $tag median=%.2f min=%.2f max=%.2f accept=%s acc/step=%s ttft=%.3f nan=%s"%(d["tok_s_median"],d["tok_s_min"],d["tok_s_max"],d["accept_ratio_median"],d["accepted_per_step_median"],d["ttft_median_s"],d["any_nan"]))
PY
}
restart_k(){ local k=$1; echo "K=$k" > $S/k.env; log "restart K=$k"; t0=$(date +%s); systemctl --user restart glm53-one-spark; for i in $(seq 1 900); do curl -sf -m 2 localhost:18080/health >/dev/null 2>&1 && break; sleep 1; done; log "ready K=$k in $(( $(date +%s)-t0 ))s"; docker logs glm53-one-spark 2>&1 | grep -iE "num_speculative_tokens|speculative" | head -3 >> $OUT/sweep.log; sleep 5; }
curl -sf -m 2 localhost:18080/health >/dev/null || { log "server not healthy at start"; exit 1; }
bench 7 baseline
for k in 4 5 6 8; do restart_k $k; bench $k sweep; done
restart_k 7; bench 7 restored
log DONE
