# training 

python train.py --max_epochs 6000 --patience 500 --obstacle_width 1.0 --obstacle_height 2.0 \
 --out_dir W_x_L_x

# testing, change the checkpoint path as needed
python eval.py --ckpt "runs/20251210_180533/checkpoint_best.pt" --n_samples 1000 --n_steps 500