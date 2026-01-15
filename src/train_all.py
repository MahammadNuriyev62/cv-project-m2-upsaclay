"""
Main training script for all models.
Trains classical ML and deep learning models for oil spill detection.
"""

import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import OilSpillDataset, get_dataloaders, get_transforms
from classical_ml import ClassicalSegmenter, load_dataset_for_classical
from deep_learning import get_model, Trainer, evaluate_model
from metrics import MetricTracker


def train_classical_models(
    root_dir: str,
    sensor: str,
    results_dir: str,
    max_train_samples: int = 500,
    sample_ratio: float = 0.03
):
    """
    Train all classical ML models.

    Args:
        root_dir: Path to dataset
        sensor: 'sentinel' or 'palsar'
        results_dir: Directory to save results
        max_train_samples: Maximum training samples to use
        sample_ratio: Pixel sampling ratio
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Classical ML Models on {sensor.upper()}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading training data...")
    train_images, train_labels = load_dataset_for_classical(
        root_dir, 'train', sensor, max_samples=max_train_samples
    )
    print(f"Loaded {len(train_images)} training images")

    print("\nLoading test data...")
    test_images, test_labels = load_dataset_for_classical(
        root_dir, 'test', sensor
    )
    print(f"Loaded {len(test_images)} test images")

    # Models to train
    classical_models = [
        ('rf', {'n_estimators': 100, 'max_depth': 15}),
        ('svm', {'C': 1.0, 'gamma': 'scale'}),
        ('knn', {'n_neighbors': 5}),
        ('gb', {'n_estimators': 100, 'max_depth': 5}),
    ]

    results = {}

    for model_type, kwargs in classical_models:
        print(f"\n{'-'*40}")
        print(f"Training {model_type.upper()}...")
        print(f"{'-'*40}")

        try:
            model = ClassicalSegmenter(model_type, patch_size=16, **kwargs)
            model.fit(train_images, train_labels, sample_ratio=sample_ratio)

            # Evaluate on test set
            print(f"\nEvaluating {model_type.upper()} on test set...")
            metric_tracker = MetricTracker()

            for img, label in tqdm(zip(test_images, test_labels), total=len(test_images)):
                pred = model.predict(img, stride=4)
                metric_tracker.update(pred, label)

            metrics = metric_tracker.get_metrics()
            results[model_type] = metrics

            print(f"\n{model_type.upper()} Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save model
            model_path = os.path.join(results_dir, f'{model_type}_{sensor}.pkl')
            model.save(model_path)
            print(f"Model saved to {model_path}")

        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}

    # Save results
    results_path = os.path.join(results_dir, f'classical_results_{sensor}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def train_deep_learning_models(
    root_dir: str,
    sensor: str,
    results_dir: str,
    epochs: int = 30,
    batch_size: int = 16
):
    """
    Train all deep learning models.

    Args:
        root_dir: Path to dataset
        sensor: 'sentinel' or 'palsar'
        results_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size
    """
    os.makedirs(results_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Training Deep Learning Models on {sensor.upper()}")
    print(f"Using device: {device}")
    print(f"{'='*60}")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir, sensor=sensor, batch_size=batch_size, num_workers=4
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Models to train
    dl_models = [
        ('unet_scratch', 'unet_scratch', None),
        ('segnet', 'segnet', None),
        ('unet_resnet34', 'unet', 'resnet34'),
        ('unet_resnet50', 'unet', 'resnet50'),
        ('deeplabv3plus_resnet34', 'deeplabv3plus', 'resnet34'),
        ('deeplabv3plus_resnet50', 'deeplabv3plus', 'resnet50'),
        ('fpn_resnet34', 'fpn', 'resnet34'),
    ]

    results = {}

    for model_name, model_type, encoder in dl_models:
        print(f"\n{'-'*40}")
        print(f"Training {model_name}...")
        print(f"{'-'*40}")

        try:
            # Create model
            if encoder:
                model = get_model(model_type, encoder_name=encoder, pretrained=True)
            else:
                model = get_model(model_type)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {num_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            # Train
            trainer = Trainer(model, device=device, learning_rate=1e-4)
            history = trainer.train(
                train_loader, val_loader,
                epochs=epochs,
                save_dir=results_dir,
                model_name=f'{model_name}_{sensor}'
            )

            # Evaluate on test set
            print(f"\nEvaluating {model_name} on test set...")
            checkpoint_path = os.path.join(results_dir, f'{model_name}_{sensor}_best.pth')
            if os.path.exists(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)

            test_metrics = evaluate_model(model, test_loader, device)

            results[model_name] = {
                'test_metrics': test_metrics,
                'best_val_iou': trainer.best_val_iou,
                'num_params': num_params,
                'trainable_params': trainable_params,
                'history': {k: [float(v) for v in vals] for k, vals in history.items()}
            }

            print(f"\n{model_name} Test Results:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # Save results
    results_path = os.path.join(results_dir, f'dl_results_{sensor}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return results


def run_all_experiments(
    root_dir: str = '/teamspace/studios/this_studio/cv-project/dataset',
    results_dir: str = '/teamspace/studios/this_studio/cv-project/results',
    sensors: list = None,
    epochs: int = 30
):
    """
    Run all experiments.

    Args:
        root_dir: Path to dataset
        results_dir: Directory to save results
        sensors: List of sensors to use
        epochs: Number of epochs for DL models
    """
    if sensors is None:
        sensors = ['sentinel', 'palsar']

    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'sensors': sensors,
        'epochs': epochs,
        'classical': {},
        'deep_learning': {}
    }

    for sensor in sensors:
        print(f"\n{'#'*60}")
        print(f"# Processing {sensor.upper()} Dataset")
        print(f"{'#'*60}")

        # Train classical models
        classical_results = train_classical_models(
            root_dir, sensor, results_dir,
            max_train_samples=500,
            sample_ratio=0.03
        )
        all_results['classical'][sensor] = classical_results

        # Train deep learning models
        dl_results = train_deep_learning_models(
            root_dir, sensor, results_dir,
            epochs=epochs, batch_size=16
        )
        all_results['deep_learning'][sensor] = dl_results

    # Save all results
    all_results_path = os.path.join(results_dir, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to {results_dir}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train oil spill detection models')
    parser.add_argument('--root_dir', type=str,
                        default='/teamspace/studios/this_studio/cv-project/dataset',
                        help='Path to dataset')
    parser.add_argument('--results_dir', type=str,
                        default='/teamspace/studios/this_studio/cv-project/results',
                        help='Directory to save results')
    parser.add_argument('--sensors', nargs='+', default=['sentinel', 'palsar'],
                        help='Sensors to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for deep learning models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'classical', 'dl'],
                        help='Training mode')

    args = parser.parse_args()

    if args.mode == 'all':
        run_all_experiments(args.root_dir, args.results_dir, args.sensors, args.epochs)
    elif args.mode == 'classical':
        for sensor in args.sensors:
            train_classical_models(args.root_dir, sensor, args.results_dir)
    elif args.mode == 'dl':
        for sensor in args.sensors:
            train_deep_learning_models(args.root_dir, sensor, args.results_dir, args.epochs)
