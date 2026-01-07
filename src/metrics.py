"""
Evaluation metrics for binary segmentation.
Includes IoU, Dice, Precision, Recall, F1, and per-class metrics.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union for binary segmentation.

    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarization

    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection

    if union == 0:
        return 1.0 if np.sum(target_binary) == 0 else 0.0

    return intersection / union


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient for binary segmentation.

    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarization

    Returns:
        Dice score
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * target_binary)
    total = np.sum(pred_binary) + np.sum(target_binary)

    if total == 0:
        return 1.0 if np.sum(target_binary) == 0 else 0.0

    return 2 * intersection / total


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute pixel-wise accuracy.

    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarization

    Returns:
        Pixel accuracy
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    correct = np.sum(pred_binary == target_binary)
    total = pred_binary.size

    return correct / total


def compute_precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarization

    Returns:
        (precision, recall, f1)
    """
    pred_binary = (pred > threshold).astype(np.float32).flatten()
    target_binary = (target > threshold).astype(np.float32).flatten()

    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all segmentation metrics.

    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarization

    Returns:
        Dictionary of all metrics
    """
    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    pixel_acc = compute_pixel_accuracy(pred, target, threshold)
    precision, recall, f1 = compute_precision_recall_f1(pred, target, threshold)

    return {
        'iou': iou,
        'dice': dice,
        'pixel_accuracy': pixel_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_confusion_matrix_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics from confusion matrix.

    Args:
        all_preds: All predicted probabilities (flattened)
        all_targets: All ground truth labels (flattened)
        threshold: Threshold for binarization

    Returns:
        Dictionary of metrics
    """
    pred_binary = (all_preds > threshold).astype(int)
    target_binary = (all_targets > threshold).astype(int)

    cm = confusion_matrix(target_binary, pred_binary, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)  # Same as recall

    return {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'specificity': specificity,
        'sensitivity': sensitivity
    }


def compute_auc_scores(
    all_preds: np.ndarray,
    all_targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute AUC scores (ROC-AUC and PR-AUC).

    Args:
        all_preds: All predicted probabilities (flattened)
        all_targets: All ground truth labels (flattened)

    Returns:
        Dictionary with AUC scores
    """
    # Sample for efficiency if too large
    if len(all_preds) > 1000000:
        indices = np.random.choice(len(all_preds), 1000000, replace=False)
        all_preds = all_preds[indices]
        all_targets = all_targets[indices]

    target_binary = (all_targets > 0.5).astype(int)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(target_binary, all_preds)
    roc_auc = auc(fpr, tpr)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(target_binary, all_preds)
    pr_auc = auc(recall, precision)

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


class MetricTracker:
    """
    Class to track and aggregate metrics over batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.all_preds = []
        self.all_targets = []
        self.running_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'pixel_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        self.num_samples = 0

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update with a batch of predictions.

        Args:
            pred: Batch of predictions (B, H, W) or (B, 1, H, W)
            target: Batch of targets (B, H, W) or (B, 1, H, W)
        """
        pred = pred.squeeze()
        target = target.squeeze()

        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]

        batch_size = pred.shape[0]

        for i in range(batch_size):
            metrics = compute_all_metrics(pred[i], target[i])
            for key in self.running_metrics:
                self.running_metrics[key] += metrics[key]
            self.num_samples += 1

            # Store for AUC computation
            self.all_preds.append(pred[i].flatten())
            self.all_targets.append(target[i].flatten())

    def get_metrics(self) -> Dict[str, float]:
        """
        Get averaged metrics.

        Returns:
            Dictionary of averaged metrics
        """
        if self.num_samples == 0:
            return self.running_metrics

        avg_metrics = {
            key: value / self.num_samples
            for key, value in self.running_metrics.items()
        }

        # Compute AUC scores
        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)

        try:
            auc_scores = compute_auc_scores(all_preds, all_targets)
            avg_metrics.update(auc_scores)
        except Exception:
            avg_metrics['roc_auc'] = 0.0
            avg_metrics['pr_auc'] = 0.0

        return avg_metrics


class DiceLoss(torch.nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(pred, target) +
            self.dice_weight * self.dice(pred, target)
        )


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    pred = np.random.rand(10, 256, 256)
    target = (np.random.rand(10, 256, 256) > 0.7).astype(np.float32)

    tracker = MetricTracker()
    tracker.update(pred, target)
    metrics = tracker.get_metrics()

    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
