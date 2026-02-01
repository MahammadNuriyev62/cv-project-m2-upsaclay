"""
Script to generate all figures and update results table for the report.
Run this after training completes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# Paths
ROOT_DIR = '/teamspace/studios/this_studio/cv-project'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')
REPORT_DIR = os.path.join(ROOT_DIR, 'report')

os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results():
    """Load results from JSON file."""
    results_path = os.path.join(RESULTS_DIR, 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def generate_metrics_comparison_figure(results, sensor='sentinel'):
    """Generate bar chart comparing all methods."""
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
        print(f"No results found for {sensor}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {'Classical': '#FF6B6B', 'Deep Learning': '#4ECDC4'}

    for ax, (metric_name, values) in zip(axes, [('IoU', ious), ('Dice', dices), ('F1', f1s)]):
        bars = ax.bar(range(len(methods)), values, color=[colors[t] for t in method_types])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Score', fontsize=12)
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    handles = [mpatches.Patch(color=colors['Classical'], label='Classical'),
               mpatches.Patch(color=colors['Deep Learning'], label='Deep Learning')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(f'Model Comparison on {sensor.upper()} Dataset', fontsize=14, y=1.08)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, f'metrics_comparison_{sensor}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_training_curves_figure(results, sensor='sentinel'):
    """Generate training curves for deep learning models."""
    if 'deep_learning' not in results or sensor not in results['deep_learning']:
        print(f"No deep learning results for {sensor}")
        return

    dl_results = results['deep_learning'][sensor]
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

    for ax, (xlabel, ylabel, title) in zip(
        axes.flatten(),
        [('Epoch', 'Loss', 'Training Loss'),
         ('Epoch', 'Loss', 'Validation Loss'),
         ('Epoch', 'IoU', 'Validation IoU'),
         ('Epoch', 'Dice', 'Validation Dice')]
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.suptitle(f'Training Curves - {sensor.upper()}', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, f'training_curves_{sensor}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_table(results):
    """Generate LaTeX results table."""
    rows = []

    for sensor in ['sentinel', 'palsar']:
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

    # Save CSV
    csv_path = os.path.join(FIGURES_DIR, 'results_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


def print_summary(results):
    """Print summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for sensor in ['sentinel', 'palsar']:
        print(f"\n{sensor.upper()} Dataset:")
        print("-" * 40)

        best_iou = 0
        best_method = ""

        if 'classical' in results and sensor in results['classical']:
            print("\nClassical Methods:")
            for method, metrics in results['classical'][sensor].items():
                if 'error' not in metrics:
                    iou = metrics.get('iou', 0)
                    dice = metrics.get('dice', 0)
                    print(f"  {method.upper():15s} IoU: {iou:.4f}  Dice: {dice:.4f}")
                    if iou > best_iou:
                        best_iou = iou
                        best_method = method

        if 'deep_learning' in results and sensor in results['deep_learning']:
            print("\nDeep Learning Methods:")
            for method, data in results['deep_learning'][sensor].items():
                if 'error' not in data and 'test_metrics' in data:
                    metrics = data['test_metrics']
                    iou = metrics.get('iou', 0)
                    dice = metrics.get('dice', 0)
                    print(f"  {method:25s} IoU: {iou:.4f}  Dice: {dice:.4f}")
                    if iou > best_iou:
                        best_iou = iou
                        best_method = method

        print(f"\nBest Method: {best_method} (IoU: {best_iou:.4f})")


def main():
    """Main function to generate all figures."""
    print("Loading results...")
    results = load_results()

    if results is None:
        print("No results found. Run training first.")
        return

    print("\nGenerating figures...")

    for sensor in ['sentinel', 'palsar']:
        generate_metrics_comparison_figure(results, sensor)
        generate_training_curves_figure(results, sensor)

    df = generate_results_table(results)
    print_summary(results)

    print("\n" + "="*60)
    print("Figure generation complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
