# COVID Detection with TDA and Swin Transformer

This project uses Topological Data Analysis (TDA) and a Swin Transformer for binary classification (COVID vs. non-COVID) on chest X-ray images.

## Pipeline

1. **Data Loading**: Images are loaded from train/val folders using torchvision's ImageFolder.
2. **TDA Feature Extraction**: Persistent homology features are extracted using giotto-tda (optional, can be used as additional features).
3. **Swin Transformer**: A Swin Transformer (from timm) is trained for binary classification.

## Requirements
- torch
- torchvision
- timm
- giotto-tda
- numpy
- matplotlib

## Usage

```bash
python main.py --data_directory /path/to/data --epochs 25
```

## Notes
- The previous CNN and ensemble logic has been removed. The pipeline is now based on TDA and Swin Transformer only.
