"""
Classical machine learning methods for oil spill detection.
Implements feature extraction and traditional classifiers.
"""

import numpy as np
import os
import pickle
from PIL import Image
from typing import Dict, List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel, gabor
from scipy import ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract handcrafted features from SAR images for oil spill detection.
    Features include: GLCM texture, statistical, edge-based, and LBP features.
    """

    def __init__(self, patch_size: int = 16):
        """
        Args:
            patch_size: Size of patches for feature extraction
        """
        self.patch_size = patch_size

    def extract_glcm_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract GLCM (Gray Level Co-occurrence Matrix) texture features.

        Args:
            gray_image: Grayscale image patch (uint8)

        Returns:
            Array of GLCM features
        """
        # Quantize to 32 levels for efficiency
        quantized = (gray_image / 8).astype(np.uint8)

        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(
            quantized,
            distances=distances,
            angles=angles,
            levels=32,
            symmetric=True,
            normed=True
        )

        # Extract properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        features = []

        for prop in properties:
            feat = graycoprops(glcm, prop)
            features.extend(feat.flatten())

        return np.array(features)

    def extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from image patch.

        Args:
            image: Image patch (can be RGB or grayscale)

        Returns:
            Array of statistical features
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        features = [
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.percentile(gray, 75) - np.percentile(gray, 25),  # IQR
            np.sum(gray > np.mean(gray)) / gray.size,  # Fraction above mean
        ]

        # Histogram features
        hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 255))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist)

        # Entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        features.append(entropy)

        return np.array(features)

    def extract_edge_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract edge-based features using Sobel filter.

        Args:
            gray_image: Grayscale image patch

        Returns:
            Array of edge features
        """
        # Sobel edge detection
        edges = sobel(gray_image.astype(np.float64) / 255.0)

        features = [
            np.mean(edges),
            np.std(edges),
            np.max(edges),
            np.sum(edges > 0.1) / edges.size,  # Edge density
        ]

        # Gradient magnitude statistics
        gx = ndimage.sobel(gray_image.astype(np.float64), axis=0)
        gy = ndimage.sobel(gray_image.astype(np.float64), axis=1)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude),
        ])

        return np.array(features)

    def extract_lbp_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern features.

        Args:
            gray_image: Grayscale image patch

        Returns:
            Array of LBP histogram features
        """
        radius = 2
        n_points = 8 * radius

        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

        # LBP histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64) / (np.sum(hist) + 1e-8)

        return hist

    def extract_gabor_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract Gabor filter features.

        Args:
            gray_image: Grayscale image patch

        Returns:
            Array of Gabor features
        """
        features = []
        frequencies = [0.1, 0.2, 0.3]
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        for freq in frequencies:
            for theta in thetas:
                try:
                    filt_real, filt_imag = gabor(
                        gray_image.astype(np.float64) / 255.0,
                        frequency=freq,
                        theta=theta
                    )
                    features.extend([
                        np.mean(filt_real),
                        np.std(filt_real),
                        np.mean(filt_imag),
                        np.std(filt_imag),
                    ])
                except Exception:
                    features.extend([0, 0, 0, 0])

        return np.array(features)

    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from an image patch.

        Args:
            image: RGB image patch

        Returns:
            Concatenated feature vector
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        features = []

        # Statistical features
        features.append(self.extract_statistical_features(image))

        # GLCM features
        features.append(self.extract_glcm_features(gray))

        # Edge features
        features.append(self.extract_edge_features(gray))

        # LBP features
        features.append(self.extract_lbp_features(gray))

        # Gabor features
        features.append(self.extract_gabor_features(gray))

        return np.concatenate(features)


def prepare_pixel_features(
    images: List[np.ndarray],
    labels: List[np.ndarray],
    patch_size: int = 16,
    sample_ratio: float = 0.1,
    balanced: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for pixel-wise classification.

    Args:
        images: List of RGB images
        labels: List of binary masks
        patch_size: Size of patches for feature extraction
        sample_ratio: Ratio of pixels to sample
        balanced: Whether to balance classes

    Returns:
        (features, labels) arrays
    """
    extractor = FeatureExtractor(patch_size)
    all_features = []
    all_labels = []

    half_patch = patch_size // 2

    for img, label in tqdm(zip(images, labels), total=len(images), desc="Extracting features"):
        h, w = label.shape[:2]

        # Pad image for border pixels
        padded_img = np.pad(
            img,
            ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
            mode='reflect'
        )

        # Sample pixels
        oil_coords = np.argwhere(label > 0.5)
        bg_coords = np.argwhere(label <= 0.5)

        if balanced and len(oil_coords) > 0:
            # Balance classes
            n_samples = min(len(oil_coords), len(bg_coords))
            n_samples = max(1, int(n_samples * sample_ratio))

            oil_indices = np.random.choice(len(oil_coords), min(n_samples, len(oil_coords)), replace=False)
            bg_indices = np.random.choice(len(bg_coords), min(n_samples, len(bg_coords)), replace=False)

            sample_coords = np.vstack([oil_coords[oil_indices], bg_coords[bg_indices]])
        else:
            # Random sampling
            all_coords = np.argwhere(np.ones_like(label, dtype=bool))
            n_samples = max(1, int(len(all_coords) * sample_ratio))
            indices = np.random.choice(len(all_coords), n_samples, replace=False)
            sample_coords = all_coords[indices]

        for coord in sample_coords:
            y, x = coord
            patch = padded_img[y:y + patch_size, x:x + patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                features = extractor.extract_all_features(patch)
                all_features.append(features)
                all_labels.append(label[y, x])

    return np.array(all_features), np.array(all_labels)


class ClassicalSegmenter:
    """
    Classical ML-based segmentation model.
    Uses patch-based feature extraction and classification.
    """

    def __init__(
        self,
        classifier_type: str = 'rf',
        patch_size: int = 16,
        **classifier_kwargs
    ):
        """
        Args:
            classifier_type: 'svm', 'rf', 'knn', or 'gb'
            patch_size: Patch size for feature extraction
            **classifier_kwargs: Additional arguments for classifier
        """
        self.classifier_type = classifier_type
        self.patch_size = patch_size
        self.extractor = FeatureExtractor(patch_size)
        self.scaler = StandardScaler()

        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                **classifier_kwargs
            )
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=classifier_kwargs.get('n_estimators', 100),
                max_depth=classifier_kwargs.get('max_depth', 15),
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
        elif classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(
                n_neighbors=classifier_kwargs.get('n_neighbors', 5),
                weights='distance',
                n_jobs=-1
            )
        elif classifier_type == 'gb':
            self.classifier = GradientBoostingClassifier(
                n_estimators=classifier_kwargs.get('n_estimators', 100),
                max_depth=classifier_kwargs.get('max_depth', 5),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, images: List[np.ndarray], labels: List[np.ndarray], sample_ratio: float = 0.05):
        """
        Train the classifier.

        Args:
            images: List of RGB images
            labels: List of binary masks
            sample_ratio: Ratio of pixels to sample for training
        """
        print(f"Preparing training features for {self.classifier_type.upper()}...")
        X, y = prepare_pixel_features(images, labels, self.patch_size, sample_ratio)

        print(f"Training set size: {len(X)} samples")
        print(f"Class distribution: Oil={np.sum(y > 0.5)}, Background={np.sum(y <= 0.5)}")

        # Binarize labels
        y = (y > 0.5).astype(int)

        # Scale features
        X = self.scaler.fit_transform(X)

        # Train classifier
        print(f"Training {self.classifier_type.upper()} classifier...")
        self.classifier.fit(X, y)

    def predict(self, image: np.ndarray, stride: int = 4) -> np.ndarray:
        """
        Predict segmentation mask for an image.

        Args:
            image: RGB image
            stride: Stride for sliding window

        Returns:
            Predicted probability map
        """
        h, w = image.shape[:2]
        half_patch = self.patch_size // 2

        # Pad image
        padded_img = np.pad(
            image,
            ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
            mode='reflect'
        )

        # Extract features for all positions
        features_list = []
        positions = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                patch = padded_img[y:y + self.patch_size, x:x + self.patch_size]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    features = self.extractor.extract_all_features(patch)
                    features_list.append(features)
                    positions.append((y, x))

        if len(features_list) == 0:
            return np.zeros((h, w), dtype=np.float32)

        # Scale and predict
        X = np.array(features_list)
        X = self.scaler.transform(X)

        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(X)[:, 1]
        else:
            probs = self.classifier.predict(X).astype(np.float32)

        # Create output map
        output = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        for (y, x), prob in zip(positions, probs):
            y_end = min(y + stride, h)
            x_end = min(x + stride, w)
            output[y:y_end, x:x_end] += prob
            count[y:y_end, x:x_end] += 1

        count = np.maximum(count, 1)
        output = output / count

        return output

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'patch_size': self.patch_size,
                'classifier_type': self.classifier_type
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ClassicalSegmenter':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['classifier_type'], data['patch_size'])
        model.classifier = data['classifier']
        model.scaler = data['scaler']
        return model


def load_dataset_for_classical(
    root_dir: str,
    split: str = 'train',
    sensor: str = 'sentinel',
    max_samples: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load dataset for classical ML methods.

    Args:
        root_dir: Path to dataset root
        split: 'train' or 'test'
        sensor: 'sentinel' or 'palsar'
        max_samples: Maximum number of samples to load

    Returns:
        (images, labels) lists
    """
    image_dir = os.path.join(root_dir, split, sensor, 'image')
    label_dir = os.path.join(root_dir, split, sensor, 'label')

    images = []
    labels = []

    filenames = sorted(os.listdir(image_dir))
    if max_samples:
        filenames = filenames[:max_samples]

    for filename in tqdm(filenames, desc=f"Loading {split} {sensor}"):
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if os.path.exists(label_path):
            img = np.array(Image.open(img_path).convert('RGB'))
            label = np.array(Image.open(label_path).convert('L'))
            label = (label > 127).astype(np.float32)

            images.append(img)
            labels.append(label)

    return images, labels


if __name__ == "__main__":
    # Test classical ML pipeline
    root = '/teamspace/studios/this_studio/cv-project/dataset'

    print("Loading data...")
    train_images, train_labels = load_dataset_for_classical(root, 'train', 'sentinel', max_samples=50)
    test_images, test_labels = load_dataset_for_classical(root, 'test', 'sentinel', max_samples=10)

    print(f"\nLoaded {len(train_images)} training images")
    print(f"Loaded {len(test_images)} test images")

    # Test Random Forest
    print("\nTraining Random Forest...")
    model = ClassicalSegmenter('rf', patch_size=16, n_estimators=50)
    model.fit(train_images, train_labels, sample_ratio=0.02)

    print("\nPredicting...")
    pred = model.predict(test_images[0])
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
