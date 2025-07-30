# COVID Detection with TDA and Rendom Forest

This project uses Topological Data Analysis (TDA) and a Classifier for binary classification (COVID vs. non-COVID) on chest X-ray images.

## Pipeline

1. **Data Loading**: Images are loaded from train/val folders using torchvision's ImageFolder.
2. **Tensor of images**: Create tensor of images then convert into numpy arrays
3. **TDA Feature Extraction**: Persistent homology features are extracted using giotto-tda (optional, can be used as additional features).
3. **Classifier**: A Random Forest(optional SVM and other classifiers) is trained for binary classification.

## Requirements
- torch
- torchvision
- giotto-tda
- numpy
- matplotlib

## Usage

```bash
python main.py --data_directory /path/to/data --epochs 10
```

