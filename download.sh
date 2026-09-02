#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO='turboderp/GLM-5.3-Flash-exl3'
TARGET_REV='51058cd551c7e570d87bd32a4adee720edce2349'
DRAFT_REPO='incoai/GLM-5.3-Flash-DFlash2'
DRAFT_REV='bf582e4eacc1810f76656d1811693ff6c6737d2a'
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/GLM-5.3-Flash-exl3-2.05bpw}"
DFLASH_DIR="${DFLASH_DIR:-$MODEL_ROOT/GLM-5.3-Flash-DFlash2}"

if [[ "${ACCEPT_DFLASH2_NC_LICENSE:-}" != 1 ]]; then
  cat >&2 <<'MSG'
DFlash2 is CC BY-NC-ND 4.0 for research/evaluation.
Read https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2 and, if acceptable, rerun with:
  ACCEPT_DFLASH2_NC_LICENSE=1 ./download.sh
MSG
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo 'Missing Hugging Face CLI. Install with: python3 -m pip install -U huggingface_hub' >&2
  exit 2
fi
mkdir -p "$MODEL_DIR" "$DFLASH_DIR"

echo "Downloading pinned target to $MODEL_DIR"
hf download "$TARGET_REPO" --revision "$TARGET_REV" --local-dir "$MODEL_DIR"
echo "Downloading pinned DFlash2 drafter to $DFLASH_DIR"
hf download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$DFLASH_DIR"

find "$MODEL_DIR" "$DFLASH_DIR" -type f \( -name '*.incomplete' -o -name '*.part' \) -print -quit | grep -q . && {
  echo 'Incomplete download artifacts remain.' >&2; exit 1;
}
printf '%s  %s\n' "$TARGET_REV" "$MODEL_DIR" > "$MODEL_DIR/ONE_SPARK_REVISION"
printf '%s  %s\n' "$DRAFT_REV" "$DFLASH_DIR" > "$DFLASH_DIR/ONE_SPARK_REVISION"
echo 'Pinned model downloads complete.'
