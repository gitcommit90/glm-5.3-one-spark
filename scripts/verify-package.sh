#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash -n start.sh download.sh scripts/serve-one-spark.sh scripts/install-systemd.sh
python3 -m py_compile overlay/*.py tests/*.py benchmarks/raw/*/*.py
python3 - <<'PY'
from pathlib import Path
required = [
 'Dockerfile','README.md','PROVENANCE.md','THIRD_PARTY_NOTICES.md','LICENSE',
 'overlay/exl3.py','overlay/patch_full_exl3_loader.py',
 'tests/test_exl3_mul1_fused_diff.py','scripts/serve-one-spark.sh',
 'benchmarks/METHODOLOGY.md'
]
missing=[x for x in required if not Path(x).is_file()]
assert not missing, missing
for p in Path('.').rglob('*'):
    if p.is_file():
        assert not any(x in p.name for x in ('.bak-', '.pre-')), p
print('package structure OK')
PY
(cd benchmarks/raw/default-sampling-pre-prefill-20260902T220641Z && sha256sum -c MANIFEST.sha256)
(cd benchmarks/raw/default-sampling-cold-prefix-20260902T221828Z && sha256sum -c MANIFEST.sha256)
echo 'Package verification passed.'
