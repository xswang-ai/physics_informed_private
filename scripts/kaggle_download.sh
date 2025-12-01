#!/usr/bin/env bash

#SBATCH --time=01:20:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file


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