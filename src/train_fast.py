"""
Fast training script for key models.
Optimized for quick experimentation while maintaining valid comparisons.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import OilSpillDataset, get_dataloaders, get_transforms
from deep_learning import get_model, Trainer, evaluate_model
from metrics import MetricTracker, compute_all_metrics


def train_simple_classical(root_dir, sensor, results_dir, max_images=200):
    """
    Train simplified classical methods using global features.
    Much faster than patch-based approach.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import pickle

    print(f"\n{'='*50}")
    print(f"Training Classical Methods on {sensor.upper()}")
    print(f"{'='*50}")

    # Load images
    image_dir = os.path.join(root_dir, 'train', sensor, 'image')
    label_dir = os.path.join(root_dir, 'train', sensor, 'label')
    filenames = sorted(os.listdir(image_dir))[:max_images]

    # Extract global features
    print("\nExtracting features...")
    features = []
    labels = []

    for fn in tqdm(filenames, desc="Processing training images"):
        img = np.array(Image.open(os.path.join(image_dir, fn)).convert('L'))
        label = np.array(Image.open(os.path.join(label_dir, fn)).convert('L'))

        # Global features per image
        feat = [
            np.mean(img), np.std(img), np.median(img),
            np.percentile(img, 25), np.percentile(img, 75),
            np.min(img), np.max(img)
        ]

        # Histogram
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 255))
        hist = hist / (np.sum(hist) + 1e-8)
        feat.extend(hist)

        features.append(feat)

        # Target: mean oil ratio (for regression-like classification)
        oil_ratio = np.mean(label > 127)
        labels.append(oil_ratio)

    X_train = np.array(features)
    y_train = np.array(labels)

    # Binarize for classification (using threshold)
    y_train_binary = (y_train > 0.1).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train_binary)

    # Evaluate on test set
    print("\nEvaluating...")
    test_image_dir = os.path.join(root_dir, 'test', sensor, 'image')
    test_label_dir = os.path.join(root_dir, 'test', sensor, 'label')
    test_filenames = sorted(os.listdir(test_image_dir))

    results = {'rf': {}, 'pixel_rf': {}}
    metric_tracker = MetricTracker()

    for fn in tqdm(test_filenames[:200], desc="Evaluating"):
        img = np.array(Image.open(os.path.join(test_image_dir, fn)).convert('L'))
        label = np.array(Image.open(os.path.join(test_label_dir, fn)).convert('L'))
        label_binary = (label > 127).astype(np.float32)

        # Extract features
        feat = [
            np.mean(img), np.std(img), np.median(img),
            np.percentile(img, 25), np.percentile(img, 75),
            np.min(img), np.max(img)
        ]
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 255))
        hist = hist / (np.sum(hist) + 1e-8)
        feat.extend(hist)

        X_test = scaler.transform([feat])
        pred_class = rf.predict(X_test)[0]
        pred_prob = rf.predict_proba(X_test)[0][1] if hasattr(rf, 'predict_proba') else pred_class

        # Create simple prediction mask (threshold-based segmentation)
        threshold = np.percentile(img, 30)  # Dark regions likely oil
        pred_mask = (img < threshold).astype(np.float32) * pred_prob

        metric_tracker.update(pred_mask, label_binary)

    results['rf'] = metric_tracker.get_metrics()
    print(f"\nRF Results: IoU={results['rf']['iou']:.4f}, Dice={results['rf']['dice']:.4f}")

    # Save
    with open(os.path.join(results_dir, f'classical_fast_{sensor}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_deep_learning_fast(root_dir, sensor, results_dir, epochs=15, batch_size=16):
    """
    Train key deep learning models.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*50}")
    print(f"Training Deep Learning Models on {sensor.upper()}")
    print(f"Device: {device}")
    print(f"{'='*50}")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir, sensor=sensor, batch_size=batch_size, num_workers=4
    )

    print(f"Train: {len(train_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    print(f"Test: {len(test_loader)} batches")

    # Models to train (key architectures)
    models_config = [
        ('unet_resnet34', 'unet', 'resnet34'),
        ('deeplabv3plus_resnet34', 'deeplabv3plus', 'resnet34'),
        ('fpn_resnet34', 'fpn', 'resnet34'),
    ]

    results = {}

    for model_name, model_type, encoder in models_config:
        print(f"\n{'-'*40}")
        print(f"Training {model_name}...")
        print(f"{'-'*40}")

        try:
            model = get_model(model_type, encoder_name=encoder, pretrained=True)

            num_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {num_params:,}")

            trainer = Trainer(model, device=device, learning_rate=1e-4)
            history = trainer.train(
                train_loader, val_loader,
                epochs=epochs,
                save_dir=results_dir,
                model_name=f'{model_name}_{sensor}'
            )

            # Evaluate on test set
            print(f"\nEvaluating {model_name}...")
            checkpoint_path = os.path.join(results_dir, f'{model_name}_{sensor}_best.pth')
            if os.path.exists(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)

            test_metrics = evaluate_model(model, test_loader, device)

            results[model_name] = {
                'test_metrics': test_metrics,
                'best_val_iou': trainer.best_val_iou,
                'num_params': num_params,
                'history': {k: [float(v) for v in vals] for k, vals in history.items()}
            }

            print(f"{model_name} Test: IoU={test_metrics['iou']:.4f}, Dice={test_metrics['dice']:.4f}")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # Save results
    with open(os.path.join(results_dir, f'dl_fast_{sensor}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return results


def run_fast_experiments(
    root_dir='/teamspace/studios/this_studio/cv-project/dataset',
    results_dir='/teamspace/studios/this_studio/cv-project/results',
    epochs=15
):
    """Run streamlined experiments."""
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'classical': {},
        'deep_learning': {}
    }

    for sensor in ['sentinel', 'palsar']:
        print(f"\n{'#'*60}")
        print(f"# Processing {sensor.upper()}")
        print(f"{'#'*60}")

        # Classical (fast version)
        classical_results = train_simple_classical(root_dir, sensor, results_dir)
        all_results['classical'][sensor] = classical_results

        # Deep learning
        dl_results = train_deep_learning_fast(root_dir, sensor, results_dir, epochs=epochs)
        all_results['deep_learning'][sensor] = dl_results

    # Save combined results
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n{'='*60}")
    print("Experiments completed!")
    print(f"Results saved to {results_dir}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    run_fast_experiments(epochs=15)
