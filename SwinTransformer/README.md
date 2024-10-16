# Rotary Position Embedding for Vision Transformer -- Swin code

This folder is RoPE Swin training code based on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) codebase.

RoPE Swin is implemented in `models/swin_transformer_rope.py`.
You can find configs from `configs/swinrope/swin_rope_*.yaml`

### Training

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main.py \
 --cfg configs/swinrope/${config_file} --data-path ${data_path} --output ${save_path} --batch-size 128
```

### Evaluation

- With pre-trained model (huggingface hub)
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 main.py \
    --eval --cfg configs/swinrope/${config_file} --resume huggingface --data-path ${data_path} --output ${save_path} --batch-size 128
    ```

- With custom `${checkpoint_file}`
    ```bash
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 main.py \
    --eval --cfg configs/swinrope/${config_file} --resume ${checkpoint_file} --data-path ${data_path} --output ${save_path} --batch-size 128
    ```