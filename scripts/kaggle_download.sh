#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-data}"                # override by passing a path arg
DATASET="ameyvarhade/physics-informed-neural-operators"

# Only run the next block once to register your API key
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  mkdir -p "$HOME/.kaggle"
  chmod 700 "$HOME/.kaggle"
  cat > "$HOME/.kaggle/kaggle.json" <<'EOF'
{"username":"xuesongwang","key":"KGAT_1752b9794e7400ae4e16595d77314c95"}
EOF
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "Saved Kaggle credentials to ~/.kaggle/kaggle.json"
fi

mkdir -p "$DEST_DIR"
kaggle datasets download -d "$DATASET" -p "$DEST_DIR" --unzip
echo "Files downloaded to $DEST_DIR"