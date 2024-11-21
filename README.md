# Towards Real-Time Practical Image Compression with Lightweight Attention
Pytorch Implementation of our paper "Towards Real Time Practical Image Compression with Lightweight Attention".

## Installation

```bash
git clone https://github.com/llsurreal919/LightweightLIC
cd LightweightLIC
pip install -U pip && pip install -e .
pip install timm
```

## Usage

### Train

Run the script for a simple training pipeline:
```bash
python examples/train.py -m tinyliclk -d /path/to/my/image/dataset/ --epochs 400 -lr 1e-4 --batch-size 8 --lambda 0.0018 --cuda --save
```

### Evaluation
Pre-trained models can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/1sSCJzXmkOSoImy2QH1KIKw?pwd=oks4) code: oks4.

An example to evaluate model:
```bash
python -m compressai.utils.eval_model checkpoint path/to/eval/data/ -a tinyliclk -p path/to/pretrained/model --cuda
```

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we add our modifications in compressai.models.tinyliclk and compressai.elan_block for usage.

The LightweightLIC model is partially built upon the [ELAN](https://github.com/xindongzhang/ELAN) and the open sourced official implementation of [TinyLIC](https://github.com/lumingzzz/TinyLIC). We thank the authors for sharing their code.

# Reference
> He Ziyang, Huang Minfeng, Luo Lei, Yang Xu and Zhu Ce, "Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection"
```
  @article{he2024towards,
    title={Towards real-time practical image compression with lightweight attention},
    author={He, Ziyang and Huang, Minfeng and Luo, Lei and Yang, Xu and Zhu, Ce},
    journal={Expert Systems with Applications},
    volume={252},
    pages={124142},
    year={2024},
    publisher={Elsevier}
  }
```
