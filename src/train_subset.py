"""
Training script using a small subset of data for faster experimentation.
Uses ~500 training images per sensor instead of the full 3000+.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import random
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import OilSpillDataset, get_transforms
from deep_learning import get_model, Trainer, evaluate_model
from metrics import MetricTracker


def get_subset_dataloaders(
    root_dir: str,
    sensor: str = 'sentinel',
    train_samples: int = 500,
    test_samples: int = 200,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Create dataloaders with limited samples for faster training.
    """
    train_transform = get_transforms('train', 256)
    test_transform = get_transforms('test', 256)

    # Full datasets
    full_train = OilSpillDataset(root_dir, split='train', sensor=sensor, transform=train_transform)
    full_test = OilSpillDataset(root_dir, split='test', sensor=sensor, transform=test_transform)

    # Create subsets
    train_indices = random.sample(range(len(full_train)), min(train_samples, len(full_train)))
    test_indices = random.sample(range(len(full_test)), min(test_samples, len(full_test)))

    # Split train into train/val (85/15)
    val_size = int(len(train_indices) * 0.15)
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    # Create validation dataset with test transforms (no augmentation)
    val_dataset_base = OilSpillDataset(root_dir, split='train', sensor=sensor, transform=test_transform)

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_dataset_base, val_indices)
    test_subset = Subset(full_test, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    return train_loader, val_loader, test_loader


def extract_texture_features(img):
    """Extract texture features from grayscale image for pixel-wise classification."""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from scipy import ndimage

    # Ensure uint8
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    h, w = img.shape
    features = np.zeros((h, w, 12), dtype=np.float32)

    # Feature 0: Original intensity (normalized)
    features[:, :, 0] = img.astype(np.float32) / 255.0

    # Feature 1-2: Local mean and std (using sliding window)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img.astype(np.float32), size=15)
    local_sq_mean = uniform_filter(img.astype(np.float32)**2, size=15)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    features[:, :, 1] = local_mean / 255.0
    features[:, :, 2] = local_std / 255.0

    # Feature 3: Deviation from local mean (dark spot indicator)
    features[:, :, 3] = (local_mean - img.astype(np.float32)) / (local_std + 1e-8)
    features[:, :, 3] = np.clip(features[:, :, 3], -3, 3) / 3.0

    # Feature 4-5: Sobel gradients
    gx = ndimage.sobel(img.astype(np.float32), axis=0)
    gy = ndimage.sobel(img.astype(np.float32), axis=1)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    features[:, :, 4] = gradient_mag / (gradient_mag.max() + 1e-8)
    features[:, :, 5] = np.arctan2(gy, gx + 1e-8) / np.pi

    # Feature 6-7: Laplacian (edge detection)
    laplacian = ndimage.laplace(img.astype(np.float32))
    features[:, :, 6] = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-8)

    # Feature 7: Local entropy approximation
    local_var = local_sq_mean - local_mean**2
    features[:, :, 7] = np.log1p(np.maximum(local_var, 0)) / 10.0

    # Feature 8-9: Multi-scale dark region detection
    dark_mask_small = uniform_filter((img < local_mean - 0.5 * local_std).astype(np.float32), size=5)
    dark_mask_large = uniform_filter((img < local_mean - 0.5 * local_std).astype(np.float32), size=25)
    features[:, :, 8] = dark_mask_small
    features[:, :, 9] = dark_mask_large

    # Feature 10-11: Percentile-based features
    p25 = np.percentile(img, 25)
    p75 = np.percentile(img, 75)
    features[:, :, 10] = (img < p25).astype(np.float32)
    features[:, :, 11] = ((img >= p25) & (img <= p75)).astype(np.float32)

    return features


def train_classical_methods(root_dir, sensor, results_dir, max_images=300, test_images=150, force_retrain=False):
    """
    Train multiple classical methods for oil spill detection.
    Methods: Otsu thresholding, K-means, SVM, Gradient Boosting

    Models are saved to results_dir and loaded if available (unless force_retrain=True).
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from skimage.filters import threshold_otsu

    print(f"\n{'='*50}")
    print(f"Training Classical Methods on {sensor.upper()}")
    print(f"{'='*50}")

    # Check for saved models
    models_path = os.path.join(results_dir, f'classical_models_{sensor}.pkl')
    if os.path.exists(models_path) and not force_retrain:
        print(f"Loading saved models from {models_path}")
        with open(models_path, 'rb') as f:
            saved_data = pickle.load(f)
        classifiers = saved_data['classifiers']
        scaler = saved_data['scaler']
        print("Models loaded successfully!")
    else:
        # Load training data
        image_dir = os.path.join(root_dir, 'train', sensor, 'image')
        label_dir = os.path.join(root_dir, 'train', sensor, 'label')
        filenames = sorted(os.listdir(image_dir))[:max_images]

        # Collect pixel samples for supervised methods
        print("\nExtracting pixel features for training...")
        all_features = []
        all_labels = []

        sample_rate = 0.02  # Sample 2% of pixels per image

        for fn in tqdm(filenames, desc="Extracting features"):
            img = np.array(Image.open(os.path.join(image_dir, fn)).convert('L'))
            label = np.array(Image.open(os.path.join(label_dir, fn)).convert('L'))
            label_binary = (label > 127).astype(np.float32)

            # Extract texture features
            feat_maps = extract_texture_features(img)
            h, w, n_feat = feat_maps.shape

            # Sample pixels (balanced)
            oil_coords = np.argwhere(label_binary > 0.5)
            bg_coords = np.argwhere(label_binary <= 0.5)

            n_oil = max(1, int(len(oil_coords) * sample_rate))
            n_bg = max(1, int(len(bg_coords) * sample_rate))
            n_samples = min(n_oil, n_bg, 500)  # Cap samples per image

            if len(oil_coords) > 0 and len(bg_coords) > 0:
                oil_idx = np.random.choice(len(oil_coords), min(n_samples, len(oil_coords)), replace=False)
                bg_idx = np.random.choice(len(bg_coords), min(n_samples, len(bg_coords)), replace=False)

                for idx in oil_idx:
                    y, x = oil_coords[idx]
                    all_features.append(feat_maps[y, x])
                    all_labels.append(1)

                for idx in bg_idx:
                    y, x = bg_coords[idx]
                    all_features.append(feat_maps[y, x])
                    all_labels.append(0)

        X_train = np.array(all_features)
        y_train = np.array(all_labels)
        print(f"Training samples: {len(X_train)} (Oil: {np.sum(y_train)}, Water: {len(y_train) - np.sum(y_train)})")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train classifiers
        classifiers = {}

        print("\nTraining SVM...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
        svm.fit(X_train_scaled, y_train)
        classifiers['svm'] = svm

        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train_scaled, y_train)
        classifiers['gb'] = gb

        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
        rf.fit(X_train_scaled, y_train)
        classifiers['rf'] = rf

        # Save trained models
        print(f"\nSaving models to {models_path}")
        with open(models_path, 'wb') as f:
            pickle.dump({'classifiers': classifiers, 'scaler': scaler}, f)
        print("Models saved successfully!")

    # Evaluate all methods
    print("\nEvaluating methods...")
    test_image_dir = os.path.join(root_dir, 'test', sensor, 'image')
    test_label_dir = os.path.join(root_dir, 'test', sensor, 'label')
    test_filenames = sorted(os.listdir(test_image_dir))[:test_images]

    # Initialize metric trackers
    metrics = {
        'otsu': MetricTracker(),
        'kmeans': MetricTracker(),
        'svm': MetricTracker(),
        'gb': MetricTracker(),
        'rf': MetricTracker()
    }

    for fn in tqdm(test_filenames, desc="Evaluating"):
        img = np.array(Image.open(os.path.join(test_image_dir, fn)).convert('L'))
        label = np.array(Image.open(os.path.join(test_label_dir, fn)).convert('L'))
        label_binary = (label > 127).astype(np.float32)

        # 1. Otsu thresholding (dark regions = oil)
        try:
            otsu_thresh = threshold_otsu(img)
            pred_otsu = (img < otsu_thresh).astype(np.float32)
        except:
            pred_otsu = (img < np.mean(img)).astype(np.float32)
        metrics['otsu'].update(pred_otsu, label_binary)

        # 2. K-means clustering
        pixels = img.flatten().reshape(-1, 1).astype(np.float32)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(pixels)
        # Assign cluster with lower mean intensity as oil
        cluster_means = [pixels[km_labels == i].mean() for i in range(2)]
        oil_cluster = np.argmin(cluster_means)
        pred_kmeans = (km_labels == oil_cluster).reshape(img.shape).astype(np.float32)
        metrics['kmeans'].update(pred_kmeans, label_binary)

        # Extract features for supervised methods
        feat_maps = extract_texture_features(img)
        h, w, n_feat = feat_maps.shape
        X_test = feat_maps.reshape(-1, n_feat)
        X_test_scaled = scaler.transform(X_test)

        # 3. SVM prediction
        pred_svm = svm.predict_proba(X_test_scaled)[:, 1].reshape(h, w)
        metrics['svm'].update(pred_svm, label_binary)

        # 4. Gradient Boosting prediction
        pred_gb = gb.predict_proba(X_test_scaled)[:, 1].reshape(h, w)
        metrics['gb'].update(pred_gb, label_binary)

        # 5. Random Forest prediction
        pred_rf = rf.predict_proba(X_test_scaled)[:, 1].reshape(h, w)
        metrics['rf'].update(pred_rf, label_binary)

    # Collect results
    results = {}
    print(f"\n{'Method':<20} {'IoU':<10} {'Dice':<10} {'F1':<10}")
    print("-" * 50)
    for name, tracker in metrics.items():
        m = tracker.get_metrics()
        results[name] = m
        print(f"{name.upper():<20} {m['iou']:.4f}     {m['dice']:.4f}     {m['f1']:.4f}")

    with open(os.path.join(results_dir, f'classical_{sensor}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_classical_simple(root_dir, sensor, results_dir, max_images=300):
    """Wrapper for backward compatibility."""
    return train_classical_methods(root_dir, sensor, results_dir, max_images)


def train_deep_learning_subset(root_dir, sensor, results_dir, epochs=20, batch_size=16,
                                train_samples=500, test_samples=200):
    """
    Train deep learning models on subset of data.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*50}")
    print(f"Training Deep Learning on {sensor.upper()}")
    print(f"Device: {device}")
    print(f"{'='*50}")

    random.seed(42)
    train_loader, val_loader, test_loader = get_subset_dataloaders(
        root_dir, sensor=sensor,
        train_samples=train_samples, test_samples=test_samples,
        batch_size=batch_size, num_workers=4
    )

    # Key models
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

            # Evaluate
            checkpoint_path = os.path.join(results_dir, f'{model_name}_{sensor}_best.pth')
            if os.path.exists(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)

            test_metrics = evaluate_model(model, test_loader, device)

            results[model_name] = {
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'best_val_iou': float(trainer.best_val_iou),
                'num_params': num_params,
                'history': {k: [float(v) for v in vals] for k, vals in history.items()}
            }

            print(f"{model_name}: IoU={test_metrics['iou']:.4f}, Dice={test_metrics['dice']:.4f}")

            # Clean up GPU memory
            del model, trainer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    with open(os.path.join(results_dir, f'dl_{sensor}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_experiments(
    root_dir='/teamspace/studios/this_studio/cv-project/dataset',
    results_dir='/teamspace/studios/this_studio/cv-project/results',
    epochs=20,
    train_samples=500,
    test_samples=200
):
    """Run experiments on subset of data."""
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': epochs,
            'train_samples': train_samples,
            'test_samples': test_samples
        },
        'classical': {},
        'deep_learning': {}
    }

    for sensor in ['sentinel', 'palsar']:
        print(f"\n{'#'*60}")
        print(f"# Processing {sensor.upper()}")
        print(f"{'#'*60}")

        # Classical
        classical_results = train_classical_simple(root_dir, sensor, results_dir)
        all_results['classical'][sensor] = classical_results

        # Deep learning
        dl_results = train_deep_learning_subset(
            root_dir, sensor, results_dir,
            epochs=epochs,
            train_samples=train_samples,
            test_samples=test_samples
        )
        all_results['deep_learning'][sensor] = dl_results

    # Save
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    run_experiments(epochs=20, train_samples=500, test_samples=200)
