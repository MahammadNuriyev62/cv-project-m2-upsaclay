# Oil Spill Detection from SAR Imagery

**Authors:** Mahammad Nuriyev & Rasul Alakbarli

Comparative analysis of classical machine learning and deep learning methods for oil spill detection using Synthetic Aperture Radar (SAR) satellite imagery.

## Dataset

Uses the [MOSD-LSAR](https://github.com/dongxr2/MOSD_LSAR) dataset (subsampled for computational efficiency):
- **Sentinel-1**: 500 training / 200 test images (256x256)
- **PALSAR**: 500 training / 200 test images (256x256)

## Results

| Type | Method | Sentinel-1 IoU | PALSAR IoU |
|------|--------|----------------|------------|
| Unsupervised | Otsu Thresholding | 0.394 | 0.256 |
| Unsupervised | K-means Clustering | 0.395 | 0.254 |
| Classical ML | SVM (texture features) | 0.502 | 0.456 |
| Classical ML | Gradient Boosting | 0.523 | 0.465 |
| Classical ML | Random Forest | 0.518 | 0.475 |
| Deep Learning | U-Net (ResNet-34) | 0.640 | 0.616 |
| Deep Learning | DeepLabV3+ (ResNet-34) | 0.647 | 0.627 |
| Deep Learning | FPN (ResNet-34) | **0.650** | **0.628** |

Deep learning methods outperform classical approaches by 24-32% relative improvement (Given that not the entire dataset was used for training, higher performance is expected with full data).

## Project Structure

```
cv-project/
├── dataset/              # SAR image dataset (sentinel/palsar)
├── src/                  # Source code
│   ├── train_subset.py   # Training (classical + deep learning)
│   ├── evaluate_models.py
│   └── generate_*.py     # Figure generation scripts
├── results/              # Trained models (.pth, .pkl)
├── figures/              # Generated visualizations
└── report/               # LaTeX report (CVPR format)
```

## Usage

```bash
# Install dependencies
pip install torch torchvision segmentation-models-pytorch albumentations scikit-learn matplotlib

# Train models
python src/train_subset.py --epochs 25 --train_samples 500

# Generate classical method predictions
python src/generate_classical_predictions.py

# Compile report
cd report && tectonic main.tex
```

## Methods

**Classical ML**: 12-dimensional texture features (intensity, local mean/std, Sobel gradients, Laplacian, multi-scale dark region detection) with SVM, Gradient Boosting, and Random Forest classifiers.

**Deep Learning**: Encoder-decoder architectures with ImageNet pre-trained ResNet-34 backbones, trained with combined BCE + Dice loss.

## License

Academic use only.
