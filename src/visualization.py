"""
Visualization utilities for oil spill detection results.
Generates figures for the report.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_sample_images(
    root_dir: str,
    output_path: str,
    sensor: str = 'sentinel',
    n_samples: int = 4
):
    """
    Plot sample images with their ground truth masks.

    Args:
        root_dir: Path to dataset
        output_path: Path to save figure
        sensor: 'sentinel' or 'palsar'
        n_samples: Number of samples to show
    """
    image_dir = os.path.join(root_dir, 'train', sensor, 'image')
    label_dir = os.path.join(root_dir, 'train', sensor, 'label')

    filenames = sorted(os.listdir(image_dir))[:n_samples * 5]  # Sample more to find diverse examples

    # Find images with oil spills
    selected = []
    for fn in filenames:
        label = np.array(Image.open(os.path.join(label_dir, fn)).convert('L'))
        oil_ratio = np.sum(label > 127) / label.size
        if 0.05 < oil_ratio < 0.5:  # Has meaningful oil spill
            selected.append(fn)
            if len(selected) >= n_samples:
                break

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    for i, fn in enumerate(selected):
        img = Image.open(os.path.join(image_dir, fn))
        label = Image.open(os.path.join(label_dir, fn)).convert('L')

        axes[0, i].imshow(img)
        axes[0, i].set_title(f'SAR Image {i+1}', fontsize=10)
        axes[0, i].axis('off')

        # Create overlay
        img_arr = np.array(img)
        label_arr = np.array(label)
        overlay = img_arr.copy()
        mask = label_arr > 127
        overlay[mask] = [255, 0, 0]  # Red for oil

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Ground Truth {i+1}', fontsize=10)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('SAR Image', fontsize=12)
    axes[1, 0].set_ylabel('With Mask', fontsize=12)

    plt.suptitle(f'{sensor.upper()} Dataset Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_class_distribution(
    root_dir: str,
    output_path: str
):
    """
    Plot class distribution for both sensors.

    Args:
        root_dir: Path to dataset
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, sensor in enumerate(['sentinel', 'palsar']):
        label_dir = os.path.join(root_dir, 'train', sensor, 'label')
        filenames = os.listdir(label_dir)

        oil_ratios = []
        for fn in filenames[:500]:  # Sample for efficiency
            label = np.array(Image.open(os.path.join(label_dir, fn)).convert('L'))
            oil_ratio = np.sum(label > 127) / label.size
            oil_ratios.append(oil_ratio)

        axes[idx].hist(oil_ratios, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        axes[idx].set_xlabel('Oil Spill Ratio per Image', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{sensor.upper()}', fontsize=12)

        # Add statistics
        mean_ratio = np.mean(oil_ratios)
        axes[idx].axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.3f}')
        axes[idx].legend()

    plt.suptitle('Class Distribution (Oil Spill Ratio)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_comparison(
    results: Dict,
    output_path: str,
    sensor: str = 'sentinel'
):
    """
    Plot bar chart comparing all methods.

    Args:
        results: Results dictionary
        output_path: Path to save figure
        sensor: 'sentinel' or 'palsar'
    """
    # Collect metrics
    methods = []
    ious = []
    dices = []
    f1s = []
    method_types = []

    # Classical methods
    if 'classical' in results and sensor in results['classical']:
        for method, metrics in results['classical'][sensor].items():
            if 'error' not in metrics:
                methods.append(method.upper())
                ious.append(metrics.get('iou', 0))
                dices.append(metrics.get('dice', 0))
                f1s.append(metrics.get('f1', 0))
                method_types.append('Classical')

    # Deep learning methods
    if 'deep_learning' in results and sensor in results['deep_learning']:
        for method, data in results['deep_learning'][sensor].items():
            if 'error' not in data and 'test_metrics' in data:
                metrics = data['test_metrics']
                methods.append(method.replace('_', '\n'))
                ious.append(metrics.get('iou', 0))
                dices.append(metrics.get('dice', 0))
                f1s.append(metrics.get('f1', 0))
                method_types.append('Deep Learning')

    if not methods:
        print("No results to plot")
        return

    # Create DataFrame
    df = pd.DataFrame({
        'Method': methods,
        'IoU': ious,
        'Dice': dices,
        'F1': f1s,
        'Type': method_types
    })

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics_to_plot = ['IoU', 'Dice', 'F1']
    colors = {'Classical': '#FF6B6B', 'Deep Learning': '#4ECDC4'}

    for ax, metric in zip(axes, metrics_to_plot):
        bars = ax.bar(range(len(methods)), df[metric], color=[colors[t] for t in method_types])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Score', fontsize=12)
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Legend
    handles = [mpatches.Patch(color=colors['Classical'], label='Classical'),
               mpatches.Patch(color=colors['Deep Learning'], label='Deep Learning')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(f'Model Comparison on {sensor.upper()} Dataset', fontsize=14, y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curves(
    results: Dict,
    output_path: str,
    sensor: str = 'sentinel'
):
    """
    Plot training curves for deep learning models.

    Args:
        results: Results dictionary
        output_path: Path to save figure
        sensor: 'sentinel' or 'palsar'
    """
    if 'deep_learning' not in results or sensor not in results['deep_learning']:
        print("No deep learning results found")
        return

    dl_results = results['deep_learning'][sensor]

    # Collect valid models with history
    valid_models = {}
    for name, data in dl_results.items():
        if 'error' not in data and 'history' in data:
            valid_models[name] = data['history']

    if not valid_models:
        print("No training history found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_models)))

    for (model_name, history), color in zip(valid_models.items(), colors):
        epochs = range(1, len(history.get('train_loss', [])) + 1)

        if 'train_loss' in history and history['train_loss']:
            axes[0, 0].plot(epochs, history['train_loss'], label=model_name, color=color)
        if 'val_loss' in history and history['val_loss']:
            axes[0, 1].plot(epochs, history['val_loss'], label=model_name, color=color)
        if 'val_iou' in history and history['val_iou']:
            axes[1, 0].plot(epochs, history['val_iou'], label=model_name, color=color)
        if 'val_dice' in history and history['val_dice']:
            axes[1, 1].plot(epochs, history['val_dice'], label=model_name, color=color)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].set_title('Validation Dice')
    axes[1, 1].legend(fontsize=7)

    plt.suptitle(f'Training Curves - {sensor.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_prediction_comparison(
    predictions: Dict[str, np.ndarray],
    image: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str
):
    """
    Plot prediction comparison across methods.

    Args:
        predictions: Dictionary of method -> prediction
        image: Original image
        ground_truth: Ground truth mask
        output_path: Path to save figure
    """
    n_methods = len(predictions)
    fig, axes = plt.subplots(2, n_methods + 2, figsize=(3 * (n_methods + 2), 6))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('SAR Image', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Ground truth
    axes[0, 1].imshow(ground_truth, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=10)
    axes[0, 1].axis('off')
    axes[1, 1].axis('off')

    # Predictions
    for i, (method, pred) in enumerate(predictions.items()):
        # Probability map
        axes[0, i + 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[0, i + 2].set_title(f'{method}\n(Probability)', fontsize=9)
        axes[0, i + 2].axis('off')

        # Binary prediction
        binary_pred = (pred > 0.5).astype(np.float32)
        axes[1, i + 2].imshow(binary_pred, cmap='gray')
        axes[1, i + 2].set_title(f'{method}\n(Binary)', fontsize=9)
        axes[1, i + 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_results_table(
    results: Dict,
    output_path: str
):
    """
    Create LaTeX table of results.

    Args:
        results: Results dictionary
        output_path: Path to save table
    """
    rows = []

    for sensor in ['sentinel', 'palsar']:
        # Classical methods
        if 'classical' in results and sensor in results['classical']:
            for method, metrics in results['classical'][sensor].items():
                if 'error' not in metrics:
                    rows.append({
                        'Sensor': sensor.upper(),
                        'Type': 'Classical',
                        'Method': method.upper(),
                        'IoU': metrics.get('iou', 0),
                        'Dice': metrics.get('dice', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1': metrics.get('f1', 0)
                    })

        # Deep learning methods
        if 'deep_learning' in results and sensor in results['deep_learning']:
            for method, data in results['deep_learning'][sensor].items():
                if 'error' not in data and 'test_metrics' in data:
                    metrics = data['test_metrics']
                    rows.append({
                        'Sensor': sensor.upper(),
                        'Type': 'Deep Learning',
                        'Method': method,
                        'IoU': metrics.get('iou', 0),
                        'Dice': metrics.get('dice', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1': metrics.get('f1', 0)
                    })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_path.replace('.tex', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate LaTeX table
    latex = df.to_latex(index=False, float_format='%.4f', escape=False)

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

    return df


def generate_all_figures(
    root_dir: str = '/teamspace/studios/this_studio/cv-project/dataset',
    results_path: str = '/teamspace/studios/this_studio/cv-project/results/all_results.json',
    figures_dir: str = '/teamspace/studios/this_studio/cv-project/figures'
):
    """
    Generate all figures for the report.

    Args:
        root_dir: Path to dataset
        results_path: Path to results JSON
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)

    # Load results
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
        print(f"Warning: Results file not found at {results_path}")

    # Dataset samples
    for sensor in ['sentinel', 'palsar']:
        plot_sample_images(
            root_dir,
            os.path.join(figures_dir, f'samples_{sensor}.png'),
            sensor=sensor
        )

    # Class distribution
    plot_class_distribution(
        root_dir,
        os.path.join(figures_dir, 'class_distribution.png')
    )

    # Metrics comparison
    if results:
        for sensor in ['sentinel', 'palsar']:
            plot_metrics_comparison(
                results,
                os.path.join(figures_dir, f'metrics_comparison_{sensor}.png'),
                sensor=sensor
            )

            plot_training_curves(
                results,
                os.path.join(figures_dir, f'training_curves_{sensor}.png'),
                sensor=sensor
            )

        # Results table
        create_results_table(
            results,
            os.path.join(figures_dir, 'results_table.tex')
        )

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    generate_all_figures()
