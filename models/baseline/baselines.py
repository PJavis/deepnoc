"""
Baseline models for NoC assignment.

1. MAC (Maximum Allele Count) — rule-based, improved version
2. Random Forest — scikit-learn, using summary features
"""

import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.constants import NUM_LOCI, NUM_FEATURES_PER_PEAK


def mac_predict(X: np.ndarray, height_threshold: float = 80.0, stutter_ratio_threshold: float = 0.15) -> np.ndarray:
    """
    MAC phiên bản nâng cao - gần với cách analyst thật làm.
    
    - Threshold cao hơn (80 rfu)
    - Lọc stutter thô: nếu peak cách ±1 repeat và height < 15% parent → bỏ qua
    - Chỉ tính distinct alleles thực sự
    """
    N = X.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    
    for i in range(N):
        max_alleles = 0
        
        for locus in range(NUM_LOCI):
            locus_data = X[i, locus]                    # [50, 89]
            heights = locus_data[:, 26] * 33000         # rfu
            alleles = locus_data[:, 24] * 100           # allele number
            
            # Lọc peak có height đủ lớn
            valid_mask = heights >= height_threshold
            if not np.any(valid_mask):
                continue
                
            valid_heights = heights[valid_mask]
            valid_alleles = alleles[valid_mask]
            
            # Lọc stutter thô (back stutter và forward stutter)
            keep = np.ones(len(valid_alleles), dtype=bool)
            
            for j in range(len(valid_alleles)):
                for k in range(len(valid_alleles)):
                    if j == k:
                        continue
                    diff = abs(valid_alleles[j] - valid_alleles[k])
                    ratio = valid_heights[j] / max(valid_heights[k], 1.0)
                    
                    # Nếu là stutter (cách 1 repeat hoặc 2 repeat) và thấp hơn parent nhiều
                    if (abs(diff - 1.0) < 0.3 or abs(diff - 2.0) < 0.3) and ratio < stutter_ratio_threshold:
                        keep[j] = False
                        break
            
            filtered_alleles = valid_alleles[keep]
            unique_alleles = np.unique(np.round(filtered_alleles, decimals=2))
            
            max_alleles = max(max_alleles, len(unique_alleles))
        
        # Final NoC
        noc_pred = math.ceil(max_alleles / 2.0)
        preds[i] = max(1, noc_pred)
    
    return preds


def extract_summary_features(X: np.ndarray) -> np.ndarray:
    """
    Extract summary features for Random Forest (giữ nguyên hoặc cải thiện nhẹ).
    """
    N = X.shape[0]
    features_list = []
    
    for i in range(N):
        feats = []
        locus_peak_counts = []
        all_heights = []
        all_alleles = []
        all_plps = []
        
        for locus in range(NUM_LOCI):
            heights = X[i, locus, :, 26] * 33000
            mask = heights > 0
            n_peaks = mask.sum()
            locus_peak_counts.append(n_peaks)
            
            if n_peaks > 0:
                all_heights.extend(heights[mask].tolist())
                alleles = X[i, locus, mask, 24] * 100
                all_alleles.extend(alleles.tolist())
                plps = X[i, locus, mask, 28]
                all_plps.extend(plps.tolist())
        
        locus_peak_counts = np.array(locus_peak_counts)
        
        # MAC (sử dụng hàm mới)
        mac = int(locus_peak_counts.max()) if len(locus_peak_counts) > 0 else 0
        feats.append(mac)
        feats.append(max(1, math.ceil(mac / 2)))           # NoC from MAC
        
        # Total peaks
        feats.append(sum(locus_peak_counts))
        
        # Per-locus peak counts
        feats.extend(locus_peak_counts.tolist())
        
        # Height statistics
        if all_heights:
            h = np.array(all_heights)
            feats.extend([h.mean(), h.std(), h.max(), h.min(), np.median(h)])
        else:
            feats.extend([0.0] * 5)
        
        # Peak label probability stats
        if all_plps:
            p = np.array(all_plps)
            feats.extend([p.mean(), p.std(), p.min()])
        else:
            feats.extend([0.0] * 3)
        
        # Mixture proportions (10 values)
        mix_props = np.zeros(10)
        for locus in range(NUM_LOCI):
            if X[i, locus, 0, 26] > 0:
                mix_props = X[i, locus, 0, 79:89]
                break
        feats.extend(mix_props.tolist())
        
        features_list.append(feats)
    
    return np.array(features_list, dtype=np.float32)


def run_mac_baseline(X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True):
    """Run improved MAC baseline."""
    preds = mac_predict(X_test, height_threshold=60.0, stutter_ratio_threshold=0.18)
    
    acc = accuracy_score(y_test, preds)
    
    if verbose:
        print(f"MAC Baseline Results:")
        print(f"  Test accuracy: {acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))   # Tắt warning
    
    return acc, preds


def train_random_forest(X_train, y_train, X_test, y_test,
                        n_estimators=500, verbose=True):
    """Random Forest (giữ nguyên logic cũ)."""
    if verbose:
        print("Extracting summary features for Random Forest...")
    
    feats_train = extract_summary_features(X_train)
    feats_test = extract_summary_features(X_test)
    
    if verbose:
        print(f"  Feature matrix: train={feats_train.shape}, test={feats_test.shape}")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    
    rf.fit(feats_train, y_train)
    
    train_preds = rf.predict(feats_train)
    test_preds = rf.predict(feats_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    if verbose:
        print(f"\nRandom Forest Results:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        print(f"\nClassification Report (test):")
        print(classification_report(y_test, test_preds, zero_division=0))
    
    return rf, train_acc, test_acc, test_preds