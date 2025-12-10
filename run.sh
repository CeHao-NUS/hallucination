# training 

python train.py --max_epochs 4000 --patience 200 

# testing, change the checkpoint path as needed
python eval.py --ckpt "runs/20251210_175945/checkpoint_best.pt" --n_samples 1000 --n_steps 500