"""
Script to evaluate trained models and generate final results.
"""

import os
import sys
import json
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import OilSpillDataset, get_transforms
from torch.utils.data import DataLoader, Subset
from deep_learning import get_model, evaluate_model
from metrics import MetricTracker


def evaluate_all_models(
    root_dir='/teamspace/studios/this_studio/cv-project/dataset',
    results_dir='/teamspace/studios/this_studio/cv-project/results',
    test_samples=200
):
    """Evaluate all trained models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load existing results
    results_path = os.path.join(results_dir, 'all_results.json')
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    models_config = [
        ('unet_resnet34', 'unet', 'resnet34'),
        ('deeplabv3plus_resnet34', 'deeplabv3plus', 'resnet34'),
        ('fpn_resnet34', 'fpn', 'resnet34'),
    ]

    for sensor in ['sentinel', 'palsar']:
        print(f"\n{'='*50}")
        print(f"Evaluating on {sensor.upper()}")
        print(f"{'='*50}")

        # Create test dataloader
        test_transform = get_transforms('test', 256)
        test_dataset = OilSpillDataset(root_dir, split='test', sensor=sensor, transform=test_transform)

        random.seed(42)
        test_indices = random.sample(range(len(test_dataset)), min(test_samples, len(test_dataset)))
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        print(f"Test samples: {len(test_subset)}")

        for model_name, model_type, encoder in models_config:
            print(f"\nEvaluating {model_name}...")

            checkpoint_path = os.path.join(results_dir, f'{model_name}_{sensor}_best.pth')

            if not os.path.exists(checkpoint_path):
                print(f"  Checkpoint not found: {checkpoint_path}")
                continue

            try:
                # Create model
                model = get_model(model_type, encoder_name=encoder, pretrained=False)
                model = model.to(device)

                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Get training history
                history = checkpoint.get('history', {})
                best_val_iou = checkpoint.get('best_val_iou', 0)

                # Evaluate
                test_metrics = evaluate_model(model, test_loader, device)

                all_results['deep_learning'][sensor][model_name] = {
                    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'best_val_iou': float(best_val_iou),
                    'num_params': sum(p.numel() for p in model.parameters()),
                    'history': {k: [float(x) for x in v] for k, v in history.items()}
                }

                print(f"  IoU: {test_metrics['iou']:.4f}")
                print(f"  Dice: {test_metrics['dice']:.4f}")
                print(f"  F1: {test_metrics['f1']:.4f}")

                # Clean up
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return all_results


if __name__ == "__main__":
    evaluate_all_models()
