
# tmux new -s hall
# Detach (leave it running):
# C-b then d
# tmux attach -t hall



#########################
# USER CONFIG
#########################

# Obstacle widths (W) and heights (H) to sweep
# WIDTHS=(0.6 0.8 1.0 1.2 1.4 1.6)
# HEIGHTS=(1.5 2.0 2.5 3.0)

WIDTHS=(0.2 0.6 1.0 1.4 1.8)
HEIGHTS=(0.5 1.0 2.0 2.5 3.5 4.0)


# Training schedule
MAX_EPOCHS=6000
PATIENCE=500

# Eval sampling
N_SAMPLES=1000
N_STEPS=500

# Where to store all runs
BASE_OUTDIR="runs_sweep_WH"
mkdir -p "${BASE_OUTDIR}"

echo "[sweep] WIDTHS = ${WIDTHS[*]}"
echo "[sweep] HEIGHTS = ${HEIGHTS[*]}"
echo "[sweep] results under ${BASE_OUTDIR}"

#########################
# TRAIN + EVAL LOOP
#########################

for H in "${HEIGHTS[@]}"; do
  for W in "${WIDTHS[@]}"; do

    RUN_NAME="W_${W}_H_${H}"
    OUT_DIR="${BASE_OUTDIR}/${RUN_NAME}"

    echo
    echo "============================================="
    echo "[sweep] Training for W=${W}, H=${H}"
    echo "         out_dir=${OUT_DIR}"
    echo "============================================="

    mkdir -p "${OUT_DIR}"

    # -------- TRAIN --------
    python train.py \
      --obstacle_width "${W}" \
      --obstacle_height "${H}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      --out_dir "${OUT_DIR}"

    CKPT_PATH="${OUT_DIR}/checkpoint_best.pt"
    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "[sweep][WARN] checkpoint not found for ${RUN_NAME}: ${CKPT_PATH}"
      continue
    fi

    echo "[sweep] Evaluating ${RUN_NAME} ..."
    python eval.py \
      --ckpt "${CKPT_PATH}" \
      --obstacle_width "${W}" \
      --obstacle_height "${H}" \
      --n_samples "${N_SAMPLES}" \
      --n_steps "${N_STEPS}" \
      --out_dir "${OUT_DIR}"

  done
done