#!/bin/bash
# Phase 1b: prose + code K sweep, K=4..8, 3 runs, temp 0, thinking off. Bench current K=7 first, then 4 5 6 8, then restore 7.
R=/home/tux/glm53-one-spark-runtime; S=$R/sweep-k; OUT=$S/results-1b-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; ln -sfn $OUT $S/results-1b.latest
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/sweep.log; }
bench(){ local k=$1 kind=$2; log "bench K=$k $kind"; BENCH_K=$k python3 $S/bench_k.py --phase $kind-k$k --kind $kind --skip-coherence --runs 3 --max-tokens 400 --out $OUT/k$k-$kind.json > $OUT/k$k-$kind.stdout 2>&1; python3 - <<PY | tee -a $OUT/sweep.log
import json;d=json.load(open("$OUT/k$k-$kind.json"));a=d["accepted_per_step_median"];t=d["tok_s_median"]
print("RESULT K=$k $kind median=%.2f min=%.2f max=%.2f accept=%s acc/step=%s step_ms=%.1f ttft=%.3f nan=%s"%(t,d["tok_s_min"],d["tok_s_max"],d["accept_ratio_median"],a,(a+1)/t*1000,d["ttft_median_s"],d["any_nan"]))
PY
}
restart_k(){ local k=$1; echo "K=$k" > $S/k.env; log "restart K=$k"; t0=$(date +%s); systemctl --user restart glm53-one-spark; for i in $(seq 1 900); do curl -sf -m 2 localhost:18080/health >/dev/null 2>&1 && break; sleep 1; done; log "ready K=$k in $(( $(date +%s)-t0 ))s"; docker logs glm53-one-spark 2>&1 | grep -o "num_spec_tokens=[0-9]*" | head -1 >> $OUT/sweep.log; sleep 5; }
for i in $(seq 1 900); do curl -sf -m 2 localhost:18080/health >/dev/null 2>&1 && break; sleep 1; done
grep -q "K=7" $S/k.env || restart_k 7
bench 7 prose; bench 7 code
for k in 4 5 6 8; do restart_k $k; bench $k prose; bench $k code; done
restart_k 7
log DONE
