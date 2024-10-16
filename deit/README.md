# Rotary Position Embedding for Vision Transformer -- DeiT code

This folder is RoPE ViT training code based on [DeiT](https://github.com/facebookresearch/deit) codebase.

RoPE ViT is implemented in `models_v2_rope.py`.
 Note that you can use other RoPE variants by simply changing model names to 

- Models
  - `rope_axial_deit_*_patch16_LS`
  - `rope_mixed_deit_*_patch16_LS`
  - `rope_axial_ape_deit_*_patch16_LS`
  - `rope_mixed_ape_deit_*_patch16_LS` 


### Training

- ViT-S
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model rope_mixed_deit_small_patch16_LS --data-path ${data_path} --output_dir ${save_path} --batch-size 256 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 4e-3 --weight-decay 0.03 --input-size 224 --drop 0.0 --drop-path 0.0 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
```

- ViT-B
  - Pretraining
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model rope_mixed_deit_base_patch16_LS --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 256 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.03 --input-size 192 --drop 0.0 --drop-path 0.1 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
  ```
  - Fine-tuning
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model rope_mixed_deit_base_patch16_LS --data-path ${data_path} --finetune ${save_path}/pretrain/checkpoint.pth --output_dir ${save_path}/finetune --batch-size 64 --epochs 20 --smoothing 0.1 --reprob 0.0 --opt adamw --lr 1e-5 --weight-decay 0.1 --input-size 224 --drop 0.0 --drop-path 0.2 --mixup 0.8 --cutmix 1.0 --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 --dist-eval
  ```


- ViT-L
  - Pretraining
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model rope_mixed_deit_large_patch16_LS  --data-path ${data_path} --output_dir ${save_path}/pretrain --batch-size 32 --epochs 400 --smoothing 0.0 --reprob 0.0 --opt fusedlamb --color-jitter 0.3 --lr 3e-3 --weight-decay 0.03 --input-size 192 --drop 0.0 --drop-path 0.4 --unscale-lr --repeated-aug --bce-loss --ThreeAugment --eval-crop-ratio 1.0 --dist-eval
  ```
  - Fine-tuning
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --use_env main.py --model rope_mixed_deit_large_patch16_LS --data-path ${data_path} --finetune ${save_path}/pretrain/checkpoint.pth --output_dir ${save_path}/finetune --batch-size 8 --epochs 20 --smoothing 0.1 --reprob 0.0 --opt adamw --lr 1e-5 --weight-decay 0.1 --input-size 224 --drop 0.0 --drop-path 0.45 --mixup 0.8 --cutmix 1.0 --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 --dist-eval
  ```

### Evaluation

- With pre-trained model (huggingface hub)
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model ${model_name} --finetune huggingface --data-path ${data_path} --output_dir ${save_path} --batch-size 128 --input-size 224 --eval --eval-crop-ratio 1.0 --dist-eval
  ```

- With custom `${checkpoint_file}`
  ```bash
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --model ${model_name} --finetune ${checkpoint_file} --data-path ${data_path} --output_dir ${save_path} --batch-size 128 --input-size 224 --eval --eval-crop-ratio 1.0 --dist-eval
  ```