# training 

python train.py --max_epochs 6000 --patience 500 

# testing, change the checkpoint path as needed
python eval.py --ckpt "runs/20251210_180533/checkpoint_best.pt" --n_samples 1000 --n_steps 500