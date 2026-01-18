"""
Generate prediction comparison figures for classical methods.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from scipy.ndimage import uniform_filter
from scipy import ndimage
from tqdm import tqdm
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def extract_texture_features(img):
    """Extract texture features from grayscale image."""
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    h, w = img.shape
    features = np.zeros((h, w, 12), dtype=np.float32)

    # Feature 0: Original intensity (normalized)
    features[:, :, 0] = img.astype(np.float32) / 255.0

    # Feature 1-2: Local mean and std
    local_mean = uniform_filter(img.astype(np.float32), size=15)
    local_sq_mean = uniform_filter(img.astype(np.float32)**2, size=15)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    features[:, :, 1] = local_mean / 255.0
    features[:, :, 2] = local_std / 255.0

    # Feature 3: Deviation from local mean
    features[:, :, 3] = (local_mean - img.astype(np.float32)) / (local_std + 1e-8)
    features[:, :, 3] = np.clip(features[:, :, 3], -3, 3) / 3.0

    # Feature 4-5: Sobel gradients
    gx = ndimage.sobel(img.astype(np.float32), axis=0)
    gy = ndimage.sobel(img.astype(np.float32), axis=1)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    features[:, :, 4] = gradient_mag / (gradient_mag.max() + 1e-8)
    features[:, :, 5] = np.arctan2(gy, gx + 1e-8) / np.pi

    # Feature 6-7: Laplacian and local entropy
    laplacian = ndimage.laplace(img.astype(np.float32))
    features[:, :, 6] = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-8)
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


def generate_classical_predictions(
    root_dir='/teamspace/studios/this_studio/cv-project/dataset',
    output_dir='/teamspace/studios/this_studio/cv-project/figures',
    results_dir='/teamspace/studios/this_studio/cv-project/results',
    max_train=150,
    n_display=4
):
    """Generate prediction figures for classical methods."""
    os.makedirs(output_dir, exist_ok=True)

    for sensor in ['sentinel', 'palsar']:
        print(f"\n{'='*50}")
        print(f"Processing {sensor.upper()}")
        print(f"{'='*50}")

        # Check for saved models
        models_path = os.path.join(results_dir, f'classical_models_{sensor}.pkl')
        if os.path.exists(models_path):
            print(f"Loading saved models from {models_path}")
            with open(models_path, 'rb') as f:
                saved_data = pickle.load(f)
            svm = saved_data['classifiers']['svm']
            gb = saved_data['classifiers']['gb']
            rf = saved_data['classifiers']['rf']
            scaler = saved_data['scaler']
            print("Models loaded successfully!")
        else:
            # Load training data for supervised methods
            train_dir = os.path.join(root_dir, 'train', sensor, 'image')
            train_label_dir = os.path.join(root_dir, 'train', sensor, 'label')
            train_files = sorted(os.listdir(train_dir))[:max_train]

            print("Extracting training features...")
            all_features = []
            all_labels = []
            sample_rate = 0.02

            for fn in tqdm(train_files, desc="Training"):
                img = np.array(Image.open(os.path.join(train_dir, fn)).convert('L'))
                label = np.array(Image.open(os.path.join(train_label_dir, fn)).convert('L'))
                label_binary = (label > 127).astype(np.float32)

                feat_maps = extract_texture_features(img)
                h, w, n_feat = feat_maps.shape

                oil_coords = np.argwhere(label_binary > 0.5)
                bg_coords = np.argwhere(label_binary <= 0.5)

                n_samples = min(300, len(oil_coords), len(bg_coords))

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
            print(f"Training samples: {len(X_train)}")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train classifiers
            print("Training classifiers...")
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
            svm.fit(X_train_scaled, y_train)

            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            gb.fit(X_train_scaled, y_train)

            rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
            rf.fit(X_train_scaled, y_train)

        # Load test images
        test_dir = os.path.join(root_dir, 'test', sensor, 'image')
        test_label_dir = os.path.join(root_dir, 'test', sensor, 'label')
        test_files = sorted(os.listdir(test_dir))

        # Select images with good oil coverage
        np.random.seed(42)
        selected_files = []
        for fn in test_files:
            label = np.array(Image.open(os.path.join(test_label_dir, fn)).convert('L'))
            oil_ratio = np.mean(label > 127)
            if 0.1 < oil_ratio < 0.5:
                selected_files.append(fn)
            if len(selected_files) >= n_display:
                break

        print(f"Generating predictions for {len(selected_files)} images...")

        # Create figure
        fig, axes = plt.subplots(n_display, 7, figsize=(21, 3*n_display))

        for i, fn in enumerate(tqdm(selected_files, desc="Predicting")):
            img = np.array(Image.open(os.path.join(test_dir, fn)).convert('L'))
            label = np.array(Image.open(os.path.join(test_label_dir, fn)).convert('L'))
            label_binary = (label > 127).astype(np.float32)

            # 1. Otsu
            try:
                otsu_thresh = threshold_otsu(img)
                pred_otsu = (img < otsu_thresh).astype(np.float32)
            except:
                pred_otsu = (img < np.mean(img)).astype(np.float32)

            # 2. K-means
            pixels = img.flatten().reshape(-1, 1).astype(np.float32)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            km_labels = kmeans.fit_predict(pixels)
            cluster_means = [pixels[km_labels == j].mean() for j in range(2)]
            oil_cluster = np.argmin(cluster_means)
            pred_kmeans = (km_labels == oil_cluster).reshape(img.shape).astype(np.float32)

            # Extract features for supervised methods
            feat_maps = extract_texture_features(img)
            h, w, n_feat = feat_maps.shape
            X_test = feat_maps.reshape(-1, n_feat)
            X_test_scaled = scaler.transform(X_test)

            # 3. SVM
            pred_svm = svm.predict_proba(X_test_scaled)[:, 1].reshape(h, w)

            # 4. GB
            pred_gb = gb.predict_proba(X_test_scaled)[:, 1].reshape(h, w)

            # 5. RF
            pred_rf = rf.predict_proba(X_test_scaled)[:, 1].reshape(h, w)

            # Plot
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('SAR Image' if i == 0 else '')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(label_binary, cmap='gray')
            axes[i, 1].set_title('Ground Truth' if i == 0 else '')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_otsu, cmap='hot')
            axes[i, 2].set_title('Otsu' if i == 0 else '')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(pred_kmeans, cmap='hot')
            axes[i, 3].set_title('K-means' if i == 0 else '')
            axes[i, 3].axis('off')

            axes[i, 4].imshow(pred_svm, cmap='hot')
            axes[i, 4].set_title('SVM' if i == 0 else '')
            axes[i, 4].axis('off')

            axes[i, 5].imshow(pred_gb, cmap='hot')
            axes[i, 5].set_title('Grad. Boost' if i == 0 else '')
            axes[i, 5].axis('off')

            axes[i, 6].imshow(pred_rf, cmap='hot')
            axes[i, 6].set_title('Random Forest' if i == 0 else '')
            axes[i, 6].axis('off')

        plt.suptitle(f'Classical Methods Predictions - {sensor.upper()}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'classical_predictions_{sensor}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: classical_predictions_{sensor}.png")


if __name__ == "__main__":
    generate_classical_predictions()
