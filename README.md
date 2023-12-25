# Towards Real Time Practical Image Compression with Lightweight Attention
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

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we add our modifications in compressai.models.tinyliclk and compressai.elan_block for usage.

The LightweightLIC model is partially built upon the [ELAN](https://github.com/xindongzhang/ELAN) and the open sourced official implementation of [TinyLIC](https://github.com/lumingzzz/TinyLIC). We thank the authors for sharing their code.