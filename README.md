<div align="center">

# Rotary Position Embedding for Vision Transformer

**[Byeongho Heo](https://sites.google.com/view/byeongho-heo/home), [Song Park](https://8uos.github.io/), [Dongyoon Han](https://sites.google.com/site/dyhan0920/), [Sangdoo Yun](https://sangdooyun.github.io/)** <br>

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

[![Paper](https://img.shields.io/badge/Paper-arxiv.2403.13298-green)](https://arxiv.org/abs/2403.13298)

</div>

Official PyTorch implementation of RoPE-ViT "Rotary Position Embedding for Vision Transformer" | [arxiv](https://arxiv.org/abs/2403.13298).

### Abstract

Rotary Position Embedding (RoPE) performs remarkably on language models, especially for length extrapolation of Transformers. However, the impacts of RoPE on computer vision domains have been underexplored, even though RoPE appears capable of enhancing Vision Transformer (ViT) performance in a way similar to the language domain. This study provides a comprehensive analysis of RoPE when applied to ViTs, utilizing practical implementations of RoPE for 2D vision data. The analysis reveals that RoPE demonstrates impressive extrapolation performance, i.e., maintaining precision while increasing image resolution at inference. It eventually leads to performance improvement for ImageNet-1k, COCO detection, and ADE-20k segmentation. We believe this study provides thorough guidelines to apply RoPE into ViT, promising improved backbone performance with minimal extra computational overhead.


## Updates

- **Mar 21, 2024**: Arxiv paper is released

## Getting Started

You can find RoPE implementations at each folder.

- `deit/`   : RoPE on DeiT-III training code *"DeiT III: Revenge of the ViT"* [original repo](https://github.com/facebookresearch/deit)
- `swin/` : RoPE on Swin Transformer training code *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"* [original repo](https://github.com/microsoft/Swin-Transformer)
- `models/` : Only RoPE model files that used for DeiT and Swin.
- `self-attn/` : Minimum implementation of RoPE in self-attention layer


## Performances

### DeiT-III

![RoPE-ViT](figures/rope_vit.png)

### Swin Transformer

![RoPE-ViT](figures/rope_swin.png)



## How to cite

```
@article{heo2024ropevit,
    title={Rotary Position Embedding for Vision Transformer},
    author={Heo, Byeongho and Park, Song and Han, Dongyoon and Yun, Sangdoo},
    year={2024},
    journal={arXiv preprint arXiv:2403.13298},
}
```

## License

This project is distributed under [Apache-2.0](LICENSE_rope-vit), <br>
except for the files below which originated from [https://github.com/meta-llama/codellama](https://github.com/meta-llama/codellama).
- [deit/models_v2_rope.py](deit/models_v2_rope.py)
- [models/swin_transformer_rope.py](models/swin_transformer_rope.py)
- [models/vit_rope.py](models/vit_rope.py)
- [self-attn/rope_self_attn.py](self-attn/rope_self_attn.py)
- [Swin-Transformer/models/swin_transformer_rope.py](Swin-Transformer/models/swin_transformer_rope.py)

```
RoPE-ViT
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```