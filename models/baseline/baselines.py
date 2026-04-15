"""
Baseline models for NoC assignment.

1. MAC (Maximum Allele Count) — rule-based, no ML
2. Random Forest — scikit-learn, using summary features
"""

import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.constants import NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK


def mac_predict(X: np.ndarray) -> np.ndarray:
    """
    Maximum Allele Count prediction.
    
    For each profile, count peaks at each locus (using non-zero height),
    find the maximum count, and assign NoC = ceil(max_count / 2).
    
    Args:
        X: [N, 24, 50, 89] input tensor
    
    Returns:
        predictions: [N] array of predicted NoC (1-indexed)
    """
    N = X.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    
    for i in range(N):
        max_alleles = 0
        for locus in range(NUM_LOCI):
            # Count peaks with non-zero height (feature index 26 = height/33000)
            heights = X[i, locus, :, 26]
            n_peaks = np.sum(heights > 0)
            max_alleles = max(max_alleles, n_peaks)
        
        preds[i] = max(1, math.ceil(max_alleles / 2))
    
    return preds


def extract_summary_features(X: np.ndarray) -> np.ndarray:
    """
    Extract summary features from the [24, 50, 89] tensor for RF/MLP baselines.
    
    Features (per profile):
    - MAC (maximum allele count across loci)
    - Total peak count
    - Per-locus peak counts (24 values)
    - Mean, std, max, min peak height
    - Mean, std peak height ratio (per locus)
    - Mean allele frequency
    - Mean peak label probability
    - Estimated mixture proportions (10 values)
    
    Returns: [N, num_features] feature matrix
    """
    N = X.shape[0]
    features_list = []
    
    for i in range(N):
        feats = []
        
        locus_peak_counts = []
        all_heights = []
        locus_height_ratios = []
        all_freqs = []
        all_plps = []
        
        for locus in range(NUM_LOCI):
            heights = X[i, locus, :, 26] * 33000  # De-normalize
            mask = heights > 0
            n_peaks = mask.sum()
            locus_peak_counts.append(n_peaks)
            
            if n_peaks > 0:
                locus_heights = heights[mask]
                all_heights.extend(locus_heights.tolist())
                
                # Height ratio: min/max within locus
                if locus_heights.max() > 0:
                    locus_height_ratios.append(
                        locus_heights.min() / locus_heights.max()
                    )
                
                # Allele frequencies (feature 27)
                freqs = X[i, locus, mask, 27]
                all_freqs.extend(freqs.tolist())
                
                # Peak label probabilities (feature 28)
                plps = X[i, locus, mask, 28]
                all_plps.extend(plps.tolist())
        
        locus_peak_counts = np.array(locus_peak_counts)
        
        # 1. MAC
        mac = int(locus_peak_counts.max()) if len(locus_peak_counts) > 0 else 0
        feats.append(mac)
        
        # 2. NoC from MAC
        feats.append(max(1, math.ceil(mac / 2)))
        
        # 3. Total peak count
        feats.append(sum(locus_peak_counts))
        
        # 4. Per-locus peak counts (24)
        feats.extend(locus_peak_counts.tolist())
        
        # 5. Mean, std, max, min of locus peak counts
        feats.append(locus_peak_counts.mean())
        feats.append(locus_peak_counts.std())
        feats.append(locus_peak_counts.max())
        feats.append(locus_peak_counts.min())
        
        # 6. Peak height statistics
        if len(all_heights) > 0:
            all_h = np.array(all_heights)
            feats.extend([all_h.mean(), all_h.std(), all_h.max(), all_h.min(),
                          np.median(all_h)])
        else:
            feats.extend([0, 0, 0, 0, 0])
        
        # 7. Height ratio statistics
        if len(locus_height_ratios) > 0:
            ratios = np.array(locus_height_ratios)
            feats.extend([ratios.mean(), ratios.std(), ratios.min()])
        else:
            feats.extend([0, 0, 0])
        
        # 8. Allele frequency statistics
        if len(all_freqs) > 0:
            freq_arr = np.array(all_freqs)
            feats.extend([freq_arr.mean(), freq_arr.std()])
        else:
            feats.extend([0, 0])
        
        # 9. Peak label probability statistics
        if len(all_plps) > 0:
            plp_arr = np.array(all_plps)
            feats.extend([plp_arr.mean(), plp_arr.std(), plp_arr.min()])
        else:
            feats.extend([0, 0, 0])
        
        # 10. Estimated mixture proportions (from features 79-88)
        # These are the same for all peaks in a profile, so take from first valid peak
        mix_props = np.zeros(10)
        for locus in range(NUM_LOCI):
            if X[i, locus, 0, 26] > 0:  # Has peaks
                mix_props = X[i, locus, 0, 79:89]
                break
        feats.extend(mix_props.tolist())
        
        features_list.append(feats)
    
    return np.array(features_list, dtype=np.float32)


def train_random_forest(X_train, y_train, X_test, y_test,
                         n_estimators=200, verbose=True):
    """
    Train a Random Forest classifier on summary features.
    
    Args:
        X_train, y_train: [N, 24, 50, 89] tensors and labels
        X_test, y_test: test data
    
    Returns:
        model, train_acc, test_acc, test_predictions
    """
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
        print(classification_report(y_test, test_preds))
    
    return rf, train_acc, test_acc, test_preds


def run_mac_baseline(X_test, y_test, verbose=True):
    """Run MAC baseline and report results."""
    preds = mac_predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    if verbose:
        print(f"MAC Baseline Results:")
        print(f"  Test accuracy: {acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds))
    
    return acc, preds