# Benchmark methodology

Canonical measurements used the loaded General23 image on one NVIDIA DGX Spark and called the vLLM backend directly at `127.0.0.1:18080`, bypassing the Spark Pulse gateway. The backend was confirmed idle before each suite. Thinking was disabled explicitly.

## Structured five-run comparison

- One excluded warmup
- Five measured requests
- 400 generated tokens each
- Temperature 0
- Decode-only timing after first streamed token
- Median: 64.052637 tok/s
- DFlash acceptance: 98.04%
- Accepted draft tokens/step: 6.863

Raw result: `raw/structured-five-run-result.json`.

## Recommended-sampling concurrency and prose

- Temperature 1.0, top-p 0.95
- Five rounds at each of C1, C2, and C4
- 400 generated tokens per stream
- Five sequential open-ended prose requests
- Metrics snapshots before and after the isolated suite

Concurrency reports three distinct quantities:

1. Per-stream decode rate measured after that stream's first token
2. Sum of per-stream decode rates, matching MiaAI's published convention
3. Strict total completions divided by submission-to-final-completion wall time

The scheduler staged C2/C4 admission; summed active-stream throughput must therefore not be represented as strict delivered wall throughput.

## Cold prefill and prefix cache

- Three independently salted cold/warm checksum pairs each at approximately 8K, 16K, and 100K prompt tokens
- Every cold request verified zero cache hits
- Warm requests verified the expected answer and nonzero prefix hits
- Four independently seeded concurrent warm 8K follow-ups
- Exactly 26 successful requests during the suite

Each raw evidence directory has a SHA-256 manifest generated over its published contents.
