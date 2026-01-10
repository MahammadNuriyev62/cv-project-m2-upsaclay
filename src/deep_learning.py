"""
Deep learning models for oil spill segmentation.
Implements U-Net, DeepLabV3+, and other architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import segmentation_models_pytorch as smp
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import os

from metrics import MetricTracker, CombinedLoss, DiceLoss


class ConvBlock(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetFromScratch(nn.Module):
    """
    U-Net implementation from scratch.
    Classic encoder-decoder architecture with skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.decoder_blocks)):
            x = self.upconvs[idx](x)

            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx](x)

        return self.final_conv(x)


class SegNet(nn.Module):
    """
    SegNet architecture with encoder-decoder and pooling indices.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 64, 2)
        self.enc2 = self._make_encoder_block(64, 128, 2)
        self.enc3 = self._make_encoder_block(128, 256, 3)
        self.enc4 = self._make_encoder_block(256, 512, 3)
        self.enc5 = self._make_encoder_block(512, 512, 3)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Decoder
        self.dec5 = self._make_decoder_block(512, 512, 3)
        self.dec4 = self._make_decoder_block(512, 256, 3)
        self.dec3 = self._make_decoder_block(256, 128, 3)
        self.dec2 = self._make_decoder_block(128, 64, 2)
        self.dec1 = self._make_decoder_block(64, 64, 2)

        self.final = nn.Conv2d(64, out_channels, 1)

    def _make_encoder_block(self, in_ch, out_ch, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_ch, out_ch, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x, idx1 = self.pool(x)
        size1 = x.size()

        x = self.enc2(x)
        x, idx2 = self.pool(x)
        size2 = x.size()

        x = self.enc3(x)
        x, idx3 = self.pool(x)
        size3 = x.size()

        x = self.enc4(x)
        x, idx4 = self.pool(x)
        size4 = x.size()

        x = self.enc5(x)
        x, idx5 = self.pool(x)

        # Decoder
        x = self.unpool(x, idx5, output_size=size4)
        x = self.dec5(x)

        x = self.unpool(x, idx4, output_size=size3)
        x = self.dec4(x)

        x = self.unpool(x, idx3, output_size=size2)
        x = self.dec3(x)

        x = self.unpool(x, idx2, output_size=size1)
        x = self.dec2(x)

        x = self.unpool(x, idx1)
        x = self.dec1(x)

        return self.final(x)


def get_model(
    model_name: str,
    encoder_name: str = 'resnet34',
    pretrained: bool = True,
    in_channels: int = 3,
    classes: int = 1
) -> nn.Module:
    """
    Get a segmentation model.

    Args:
        model_name: 'unet', 'unet_scratch', 'deeplabv3plus', 'segnet', 'fpn', 'pspnet'
        encoder_name: Encoder backbone for pretrained models
        pretrained: Whether to use pretrained encoder weights
        in_channels: Number of input channels
        classes: Number of output classes

    Returns:
        PyTorch model
    """
    encoder_weights = 'imagenet' if pretrained else None

    if model_name == 'unet':
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_name == 'unet_scratch':
        return UNetFromScratch(in_channels, classes)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_name == 'segnet':
        return SegNet(in_channels, classes)
    elif model_name == 'fpn':
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_name == 'pspnet':
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_name == 'manet':
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class Trainer:
    """
    Trainer class for deep learning segmentation models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.scheduler = None
        self.best_val_iou = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        metric_tracker = MetricTracker()

        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            # Convert to numpy for metrics
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()

            metric_tracker.update(preds, targets)

        metrics = metric_tracker.get_metrics()
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_dir: str = 'models',
        model_name: str = 'model'
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save models
            model_name: Model name for saving

        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_dice'].append(val_metrics['dice'])

            # Update scheduler
            self.scheduler.step()

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")

            # Save best model
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                save_path = os.path.join(save_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_iou': self.best_val_iou,
                    'history': self.history
                }, save_path)
                print(f"  Saved best model (IoU: {self.best_val_iou:.4f})")

        # Save final model
        final_path = os.path.join(save_dir, f'{model_name}_final.pth')
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou,
            'history': self.history
        }, final_path)

        return self.history

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
        self.history = checkpoint.get('history', self.history)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """
        Predict segmentation mask for a single image.

        Args:
            image: Input tensor (1, C, H, W) or (C, H, W)

        Returns:
            Predicted probability map
        """
        self.model.eval()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        output = self.model(image)
        pred = torch.sigmoid(output).cpu().numpy()

        return pred.squeeze()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate a model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)

    metric_tracker = MetricTracker()
    criterion = CombinedLoss()
    total_loss = 0.0

    for images, masks in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = torch.sigmoid(outputs).cpu().numpy()
        targets = masks.cpu().numpy()

        metric_tracker.update(preds, targets)

    metrics = metric_tracker.get_metrics()
    metrics['loss'] = total_loss / len(test_loader)

    return metrics


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...")

    x = torch.randn(2, 3, 256, 256)

    # Test U-Net from scratch
    model = UNetFromScratch(3, 1)
    out = model(x)
    print(f"U-Net (scratch) output shape: {out.shape}")

    # Test SegNet
    model = SegNet(3, 1)
    out = model(x)
    print(f"SegNet output shape: {out.shape}")

    # Test pretrained models
    for name in ['unet', 'deeplabv3plus', 'fpn']:
        model = get_model(name)
        out = model(x)
        print(f"{name} output shape: {out.shape}")

    print("\nAll models tested successfully!")
