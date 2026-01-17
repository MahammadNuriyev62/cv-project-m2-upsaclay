"""
Generate prediction comparison figures for the report.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import OilSpillDataset, get_transforms
from deep_learning import get_model


def generate_prediction_figures(
    root_dir='/teamspace/studios/this_studio/cv-project/dataset',
    results_dir='/teamspace/studios/this_studio/cv-project/results',
    figures_dir='/teamspace/studios/this_studio/cv-project/figures',
    sensor='sentinel',
    n_samples=4
):
    """Generate prediction comparison figures."""
    os.makedirs(figures_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test dataset without transforms for visualization
    test_transform = get_transforms('test', 256)
    test_dataset = OilSpillDataset(root_dir, split='test', sensor=sensor, transform=test_transform, return_path=True)

    # Find good samples with oil spills
    random.seed(42)
    good_indices = []
    for i in random.sample(range(len(test_dataset)), min(100, len(test_dataset))):
        _, mask, _ = test_dataset[i]
        oil_ratio = mask.sum().item() / mask.numel()
        if 0.05 < oil_ratio < 0.4:
            good_indices.append(i)
            if len(good_indices) >= n_samples:
                break

    # Load models
    models = {}
    model_configs = [
        ('U-Net', 'unet_resnet34', 'unet', 'resnet34'),
        ('DeepLabV3+', 'deeplabv3plus_resnet34', 'deeplabv3plus', 'resnet34'),
        ('FPN', 'fpn_resnet34', 'fpn', 'resnet34'),
    ]

    for display_name, model_name, model_type, encoder in model_configs:
        checkpoint_path = os.path.join(results_dir, f'{model_name}_{sensor}_best.pth')
        if os.path.exists(checkpoint_path):
            model = get_model(model_type, encoder_name=encoder, pretrained=False)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            models[display_name] = model
            print(f"Loaded {display_name}")

    if not models:
        print("No models found!")
        return

    # Generate predictions
    fig, axes = plt.subplots(n_samples, len(models) + 2, figsize=(3 * (len(models) + 2), 3 * n_samples))

    for row, idx in enumerate(good_indices):
        img_tensor, mask, img_path = test_dataset[idx]

        # Load original image for display
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        mask_np = mask.squeeze().numpy()

        # Show original image
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title('SAR Image' if row == 0 else '', fontsize=10)
        axes[row, 0].axis('off')

        # Show ground truth
        axes[row, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '', fontsize=10)
        axes[row, 1].axis('off')

        # Show predictions
        for col, (model_name, model) in enumerate(models.items()):
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0).to(device))
                pred = torch.sigmoid(pred).cpu().numpy().squeeze()

            axes[row, col + 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[row, col + 2].set_title(model_name if row == 0 else '', fontsize=10)
            axes[row, col + 2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(figures_dir, f'predictions_{sensor}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Generate overlay figure
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))

    for row, idx in enumerate(good_indices):
        img_tensor, mask, img_path = test_dataset[idx]
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        mask_np = mask.squeeze().numpy()

        # Original
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title('SAR Image' if row == 0 else '', fontsize=11)
        axes[row, 0].axis('off')

        # Ground truth overlay
        overlay_gt = orig_img.copy().astype(float) / 255
        overlay_gt[mask_np > 0.5, 0] = 1.0  # Red channel for GT
        overlay_gt[mask_np > 0.5, 1] *= 0.3
        overlay_gt[mask_np > 0.5, 2] *= 0.3
        axes[row, 1].imshow(overlay_gt)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '', fontsize=11)
        axes[row, 1].axis('off')

        # Best model prediction (FPN)
        if 'FPN' in models:
            with torch.no_grad():
                pred = models['FPN'](img_tensor.unsqueeze(0).to(device))
                pred = torch.sigmoid(pred).cpu().numpy().squeeze()

            # Prediction heatmap
            axes[row, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[row, 2].set_title('FPN Prediction' if row == 0 else '', fontsize=11)
            axes[row, 2].axis('off')

            # Prediction overlay
            pred_binary = pred > 0.5
            overlay_pred = orig_img.copy().astype(float) / 255
            overlay_pred[pred_binary, 1] = 1.0  # Green for prediction
            overlay_pred[pred_binary, 0] *= 0.3
            overlay_pred[pred_binary, 2] *= 0.3
            axes[row, 3].imshow(overlay_pred)
            axes[row, 3].set_title('FPN Overlay' if row == 0 else '', fontsize=11)
            axes[row, 3].axis('off')

    plt.tight_layout()
    output_path = os.path.join(figures_dir, f'predictions_overlay_{sensor}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for sensor in ['sentinel', 'palsar']:
        print(f"\nGenerating predictions for {sensor}...")
        generate_prediction_figures(sensor=sensor, n_samples=4)
