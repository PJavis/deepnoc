"""
Data loader for PROVEDIt GeneMapper-filtered CSV files.

Reads CSV/XLSX files, extracts peak information per locus,
and builds the [24 × 50 × 89] input tensors for deepNoC.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.constants import (
    GLOBALFILER_LOCI, LOCUS_TO_IDX, LOCUS_ALIASES,
    NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK, MAX_NOC,
    ALLELE_NORM, SIZE_NORM, HEIGHT_NORM,
    LOCUS_PEAK_NORM, PROFILE_PEAK_NORM,
    DEFAULT_ALLELE_FREQ,
    EXPECTED_BACK_STUTTER_RATIO, EXPECTED_DBL_BACK_STUTTER_RATIO,
    EXPECTED_FWD_STUTTER_RATIO, EXPECTED_PT2_STUTTER_RATIO,
)


def normalize_locus_name(name: str) -> Optional[str]:
    """Map various locus name formats to our canonical names."""
    name = name.strip()
    if name in LOCUS_TO_IDX:
        return name
    if name in LOCUS_ALIASES:
        return LOCUS_ALIASES[name]
    # Try case-insensitive match
    for locus in GLOBALFILER_LOCI:
        if name.lower() == locus.lower():
            return locus
    return None


def parse_noc_from_filename(filename: str) -> int:
    """
    Extract the number of contributors from a PROVEDIt filename or path.
    
    Examples:
        '1-Person/...' -> 1
        '2-5-Persons/...' with sample info -> parsed from sample
        'RD12-0002_25sec_GM_F_3P.csv' -> 3
    """
    basename = os.path.basename(filename)
    
    # Try pattern like _1P.csv, _2P.csv, _3P.csv
    match = re.search(r'_(\d)P\.', basename)
    if match:
        return int(match.group(1))

    # Multi-NoC files like _2-5P.csv need sample-level parsing.
    if re.search(r'_\d-\dP\.', basename):
        return -1
    
    # Try from directory path: '1-Person', '2-Person', etc.
    parts = filename.replace('\\', '/').split('/')
    for part in parts:
        match = re.match(r'^(\d)-Person', part)
        if match:
            return int(match.group(1))
    
    return -1  # Unknown


def parse_noc_from_sample_name(sample_name: str, filepath: str) -> int:
    """
    Determine the NoC for a specific sample within a multi-person file.
    
    For files like '2-5P', each sample row needs individual NoC assignment
    based on sample name patterns.
    """
    # First try filename-level
    file_noc = parse_noc_from_filename(filepath)
    if file_noc > 0 and file_noc <= 5:
        # For single-NoC files (1P, 2P, etc.), all samples have same NoC
        return file_noc
    
    # For multi-NoC files (2-5P), parse from sample name
    # PROVEDIt naming: typically contains mixture ratios or person counts
    # e.g., "1to1_..." = 2 persons, "1to1to1_..." = 3 persons
    sample = str(sample_name).lower()

    # PROVEDIt filtered files often encode the contributor ratios between
    # hyphens, e.g. "...-1;1-..." (2 contributors) or "...-1;2;1-..." (3).
    ratio_match = re.search(r'-(\d+(?:;\d+)+)-', sample)
    if ratio_match:
        ratio_terms = ratio_match.group(1).split(';')
        noc = len(ratio_terms)
        if 1 <= noc <= MAX_NOC:
            return noc
    
    # Count "to" separators in ratio strings
    to_count = sample.count('to')
    if to_count > 0:
        return to_count + 1
    
    # Count number of contributors from mixture notation
    # e.g., "2p", "3p", "4p", "5p" somewhere in the sample name
    match = re.search(r'(\d)\s*p(?:erson|er|$|\s)', sample)
    if match:
        return int(match.group(1))
    
    return file_noc if file_noc > 0 else -1


def read_genemapper_csv(filepath: str) -> pd.DataFrame:
    """
    Read a GeneMapper-format CSV or XLSX file.
    
    Returns a DataFrame with standardized columns:
    - SampleName, Marker, Dye, Allele, Size, Height
    
    GeneMapper CSV typically has wide format with 
    Allele 1, Allele 2, ..., Size 1, Size 2, ..., Height 1, Height 2, ...
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == '.xlsx':
        df = pd.read_excel(filepath, engine='openpyxl')
    elif ext == '.xls':
        df = pd.read_excel(filepath, engine='xlrd')
    else:
        # Try different separators
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            if len(df.columns) < 3:
                df = pd.read_csv(filepath, sep=',', low_memory=False)
        except Exception:
            df = pd.read_csv(filepath, sep=',', low_memory=False)
    
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Identify the wide-format allele/size/height columns
    allele_cols = sorted([c for c in df.columns if re.match(r'Allele\s*\d+', c, re.I)],
                         key=lambda x: int(re.search(r'\d+', x).group()))
    size_cols = sorted([c for c in df.columns if re.match(r'Size\s*\d+', c, re.I)],
                       key=lambda x: int(re.search(r'\d+', x).group()))
    height_cols = sorted([c for c in df.columns if re.match(r'Height\s*\d+', c, re.I)],
                         key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Find the sample name and marker columns
    sample_col = None
    for candidate in ['Sample Name', 'SampleName', 'Sample File', 'Sample']:
        if candidate in df.columns:
            sample_col = candidate
            break
    
    marker_col = None
    for candidate in ['Marker', 'Locus', 'marker', 'locus']:
        if candidate in df.columns:
            marker_col = candidate
            break
    
    dye_col = None
    for candidate in ['Dye', 'dye', 'Color', 'Dye Color']:
        if candidate in df.columns:
            dye_col = candidate
            break
    
    if sample_col is None or marker_col is None:
        raise ValueError(f"Cannot find Sample/Marker columns in {filepath}. "
                         f"Found columns: {list(df.columns)}")
    
    # Melt wide format to long format (one row per peak)
    records = []
    for _, row in df.iterrows():
        sample = row[sample_col]
        marker = row[marker_col]
        dye = row[dye_col] if dye_col else ""
        
        # Normalize locus name
        norm_marker = normalize_locus_name(str(marker))
        if norm_marker is None:
            continue  # Skip non-GlobalFiler loci
        
        for i in range(len(allele_cols)):
            allele_val = row[allele_cols[i]] if i < len(allele_cols) else None
            size_val = row[size_cols[i]] if i < len(size_cols) else None
            height_val = row[height_cols[i]] if i < len(height_cols) else None
            
            # Skip empty/NaN peaks
            if pd.isna(allele_val) or str(allele_val).strip() == '':
                continue
            
            # Convert allele to float (handle 'X', 'Y', 'OL' etc.)
            try:
                allele_num = float(allele_val)
            except (ValueError, TypeError):
                # Handle non-numeric alleles (AMEL: X=1, Y=2; OL=off-ladder)
                allele_str = str(allele_val).upper()
                if allele_str == 'X':
                    allele_num = 1.0
                elif allele_str == 'Y':
                    allele_num = 2.0
                elif allele_str == 'OL':
                    continue  # Skip off-ladder
                else:
                    continue
            
            try:
                size_num = float(size_val) if not pd.isna(size_val) else 0.0
            except (ValueError, TypeError):
                size_num = 0.0
            
            try:
                height_num = float(height_val) if not pd.isna(height_val) else 0.0
            except (ValueError, TypeError):
                height_num = 0.0
            
            if height_num <= 0:
                continue  # Skip zero-height peaks
            
            records.append({
                'SampleName': sample,
                'Marker': norm_marker,
                'Dye': dye,
                'Allele': allele_num,
                'Size': size_num,
                'Height': height_num,
            })
    
    return pd.DataFrame(records)


def detect_stutter_relationships(peaks_df: pd.DataFrame, locus: str) -> dict:
    """
    For peaks at a given locus, detect stutter relationships.
    
    Returns dict mapping peak_idx -> {stutter_type: parent_peak_idx}
    and parent_idx -> {stutter_type: stutter_peak_idx}
    """
    peaks = peaks_df.sort_values('Allele').reset_index(drop=True)
    n = len(peaks)
    
    stutter_info = {i: {} for i in range(n)}
    parent_info = {i: {} for i in range(n)}
    
    for i in range(n):
        allele_i = peaks.iloc[i]['Allele']
        height_i = peaks.iloc[i]['Height']
        
        for j in range(n):
            if i == j:
                continue
            allele_j = peaks.iloc[j]['Allele']
            height_j = peaks.iloc[j]['Height']
            diff = allele_j - allele_i
            
            # Back stutter: peak is -1 repeat from parent
            if abs(diff - 1.0) < 0.01 and height_i < height_j:
                stutter_info[i]['back'] = j       # i is back stutter of j
                parent_info[j]['back'] = i         # j is parent of back stutter i
            
            # Double back stutter: peak is -2 repeats from parent
            elif abs(diff - 2.0) < 0.01 and height_i < height_j:
                stutter_info[i]['dbl_back'] = j
                parent_info[j]['dbl_back'] = i
            
            # Forward stutter: peak is +1 repeat from parent
            elif abs(diff + 1.0) < 0.01 and height_i < height_j:
                stutter_info[i]['forward'] = j
                parent_info[j]['forward'] = i
            
            # 0.2 repeat stutter
            elif abs(abs(diff) - 0.2) < 0.05 and height_i < height_j:
                stutter_info[i]['pt2'] = j
                parent_info[j]['pt2'] = i
    
    return stutter_info, parent_info


def build_peak_features(peaks_df: pd.DataFrame, locus: str,
                        total_peaks_in_profile: int) -> np.ndarray:
    """
    Build the 89-feature vector for each peak at a given locus.
    
    Returns: array of shape [num_peaks, 89]
    """
    peaks = peaks_df.sort_values('Size').reset_index(drop=True)
    n = len(peaks)
    
    if n == 0:
        return np.zeros((0, NUM_FEATURES_PER_PEAK))
    
    # Detect stutter relationships
    stutter_info, parent_info = detect_stutter_relationships(peaks, locus)
    
    features = np.zeros((n, NUM_FEATURES_PER_PEAK))
    
    locus_idx = LOCUS_TO_IDX.get(locus, 0)
    
    for i in range(n):
        row = peaks.iloc[i]
        allele = row['Allele']
        size = row['Size']
        height = row['Height']
        
        # 1-24: One-hot encoded locus
        features[i, locus_idx] = 1.0
        
        # 25: Allele designation / 100
        features[i, 24] = allele / ALLELE_NORM
        
        # 26: Size in base pairs / 100
        features[i, 25] = size / SIZE_NORM
        
        # 27: Height in rfu / 33000
        features[i, 26] = height / HEIGHT_NORM
        
        # 28: Allele frequency (using default - replace with actual freq table)
        features[i, 27] = DEFAULT_ALLELE_FREQ
        
        # 29: Peak label probability
        # Without the MHCNN, we use a heuristic: 
        # higher peaks more likely allelic, stutter peaks lower
        features[i, 28] = estimate_peak_label_probability(
            height, peaks['Height'].max(), i in stutter_info and len(stutter_info[i]) > 0
        )
        
        # 30-77: Stutter information (4 types × 6 values × 2 directions = 48 values)
        # If peak is stutter of parent (features 30-53)
        stutter_types = ['back', 'dbl_back', 'forward', 'pt2']
        expected_ratios = [
            EXPECTED_BACK_STUTTER_RATIO,
            EXPECTED_DBL_BACK_STUTTER_RATIO,
            EXPECTED_FWD_STUTTER_RATIO,
            EXPECTED_PT2_STUTTER_RATIO,
        ]
        
        for st_idx, (st_type, exp_ratio) in enumerate(zip(stutter_types, expected_ratios)):
            base = 29 + st_idx * 6  # 29, 35, 41, 47
            if st_type in stutter_info.get(i, {}):
                parent_idx = stutter_info[i][st_type]
                parent_row = peaks.iloc[parent_idx]
                features[i, base + 0] = parent_row['Allele'] / ALLELE_NORM
                features[i, base + 1] = parent_row['Height'] / HEIGHT_NORM
                features[i, base + 2] = min(height / max(parent_row['Height'], 1), 1.0)
                features[i, base + 3] = exp_ratio
                features[i, base + 4] = DEFAULT_ALLELE_FREQ  # parent freq
                features[i, base + 5] = estimate_peak_label_probability(
                    parent_row['Height'], peaks['Height'].max(), False
                )
        
        # If peak is parent of stutter (features 54-77)
        for st_idx, (st_type, exp_ratio) in enumerate(zip(stutter_types, expected_ratios)):
            base = 53 + st_idx * 6  # 53, 59, 65, 71
            if st_type in parent_info.get(i, {}):
                stutter_idx = parent_info[i][st_type]
                stutter_row = peaks.iloc[stutter_idx]
                features[i, base + 0] = stutter_row['Allele'] / ALLELE_NORM
                features[i, base + 1] = stutter_row['Height'] / HEIGHT_NORM
                features[i, base + 2] = min(stutter_row['Height'] / max(height, 1), 1.0)
                features[i, base + 3] = exp_ratio
                features[i, base + 4] = DEFAULT_ALLELE_FREQ  # stutter freq
                features[i, base + 5] = estimate_peak_label_probability(
                    stutter_row['Height'], peaks['Height'].max(),
                    True
                )
        
        # 78: Total peaks at locus / 100
        features[i, 77] = n / LOCUS_PEAK_NORM
        
        # 79: Total peaks in profile / 1000
        features[i, 78] = total_peaks_in_profile / PROFILE_PEAK_NORM
        
        # 80-89: Estimated mixture proportions (10 contributors)
        # Will be filled at the profile level later
        # (features[i, 79:89] = 0 for now, filled by smart_start)
    
    return features


def estimate_peak_label_probability(height: float, max_height: float,
                                     is_likely_stutter: bool) -> float:
    """
    Heuristic estimate of peak label probability (probability of being non-artefactual).
    
    Without the MHCNN from Taylor et al. [9], we use a simple heuristic:
    - Higher relative peaks are more likely allelic
    - Known stutter positions have lower probability
    """
    if max_height <= 0:
        return 0.5
    
    relative_height = height / max_height
    
    if is_likely_stutter:
        # Stutter peaks: lower plp
        plp = 0.3 + 0.3 * relative_height
    else:
        # Non-stutter: higher plp
        plp = 0.6 + 0.4 * relative_height
    
    return min(max(plp, 0.01), 0.99)


def estimate_smart_start(profile_peaks: dict, max_noc: int = 10) -> np.ndarray:
    """
    Simplified version of the 'smart start' algorithm from STRmix (Appendix 1 of [15]).
    
    Estimates mixture proportions for up to max_noc contributors based on peak heights.
    
    Returns: array of shape [max_noc] summing to 1.
    """
    # Collect all peak heights across loci
    all_heights = []
    for locus, peaks in profile_peaks.items():
        if len(peaks) > 0:
            all_heights.extend(peaks['Height'].tolist())
    
    if len(all_heights) == 0:
        return np.ones(max_noc) / max_noc
    
    # Sort heights descending
    all_heights = sorted(all_heights, reverse=True)
    total = sum(all_heights)
    
    if total == 0:
        return np.ones(max_noc) / max_noc
    
    # Estimate: assign top peaks to contributors in decreasing order
    # This is a simplified version - the real smart start is more sophisticated
    proportions = np.zeros(max_noc)
    
    # Use peak height distribution to estimate contributor proportions
    # Group peaks into rough contributor clusters
    n_peaks = len(all_heights)
    chunk_size = max(1, n_peaks // max_noc)
    
    for c in range(max_noc):
        start = c * chunk_size
        end = min(start + chunk_size, n_peaks)
        if start < n_peaks:
            proportions[c] = sum(all_heights[start:end])
        else:
            proportions[c] = 0.001  # Small default for superfluous contributors
    
    # Normalize
    proportions = proportions / proportions.sum()
    
    # Sort descending (largest contributor first)
    proportions = np.sort(proportions)[::-1]
    
    return proportions


def build_profile_tensor(sample_peaks: pd.DataFrame,
                          total_peaks_in_profile: int) -> np.ndarray:
    """
    Build the [24 × 50 × 89] input tensor for a single DNA profile.
    
    Args:
        sample_peaks: DataFrame with columns [Marker, Allele, Size, Height, ...]
        total_peaks_in_profile: total number of peaks across all loci
    
    Returns:
        tensor of shape [24, 50, 89]
    """
    tensor = np.zeros((NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK),
                      dtype=np.float32)
    
    # Group peaks by locus
    profile_peaks_by_locus = {}
    for locus in GLOBALFILER_LOCI:
        locus_peaks = sample_peaks[sample_peaks['Marker'] == locus]
        profile_peaks_by_locus[locus] = locus_peaks
    
    # Estimate mixture proportions using smart start
    mix_props = estimate_smart_start(profile_peaks_by_locus)
    
    # Build features for each locus
    for locus_idx, locus in enumerate(GLOBALFILER_LOCI):
        locus_peaks = profile_peaks_by_locus[locus]
        
        if len(locus_peaks) == 0:
            continue
        
        features = build_peak_features(locus_peaks, locus, total_peaks_in_profile)
        n_peaks = min(len(features), MAX_PEAKS_PER_LOCUS)
        
        tensor[locus_idx, :n_peaks, :] = features[:n_peaks]
        
        # Fill mixture proportion features (80-89) for all peaks at this locus
        tensor[locus_idx, :n_peaks, 79:89] = mix_props
    
    return tensor


def load_provedit_dataset(data_dir: str,
                           kit_filter: str = "GF",
                           injection_filter: str = "25sec",
                           instrument_filter: str = "3500",
                           verbose: bool = True) -> tuple:
    """
    Load and process PROVEDIt dataset from filtered CSV files.
    
    Args:
        data_dir: Path to 'data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered'
        kit_filter: Filter for profiling kit (e.g., 'GF' for GlobalFiler)
        injection_filter: Filter for injection time (e.g., '25sec')
        instrument_filter: Filter for instrument (e.g., '3500')
        verbose: Print progress
    
    Returns:
        X: array of shape [N, 24, 50, 89]
        y: array of shape [N] (NoC labels, 1-indexed)
        sample_names: list of sample identifiers
    """
    # Find matching subdirectory
    base_dir = Path(data_dir)
    target_dirs = list(base_dir.glob(f"*{instrument_filter}*{kit_filter}*"))
    
    if not target_dirs:
        # Try broader search
        target_dirs = list(base_dir.glob("*"))
        target_dirs = [d for d in target_dirs if d.is_dir() 
                       and instrument_filter in d.name and kit_filter in d.name]
    
    if not target_dirs:
        raise FileNotFoundError(
            f"No matching directory found in {data_dir} for "
            f"instrument={instrument_filter}, kit={kit_filter}"
        )
    
    target_dir = target_dirs[0]
    if verbose:
        print(f"Loading from: {target_dir}")
    
    # Find all CSV/XLSX files with matching injection time
    data_files = []
    for ext in ['*.csv', '*.xlsx']:
        data_files.extend(target_dir.rglob(ext))
    
    # Filter for injection time and exclude known genotype files
    data_files = [
        f for f in data_files
        if injection_filter in f.name
        and 'Known Genotype' not in f.name
        and 'Genotype' not in f.name
        and not f.name.startswith('~$')  # Skip temp files
    ]
    
    if verbose:
        print(f"Found {len(data_files)} data files")
    
    all_X = []
    all_y = []
    all_names = []
    
    for fpath in sorted(data_files):
        if verbose:
            print(f"  Processing: {fpath.name}")
        
        try:
            df = read_genemapper_csv(str(fpath))
        except Exception as e:
            print(f"  WARNING: Failed to read {fpath.name}: {e}")
            continue
        
        if len(df) == 0:
            print(f"  WARNING: No valid peaks in {fpath.name}")
            continue
        
        # Get unique samples in this file
        samples = df['SampleName'].unique()
        
        for sample in samples:
            sample_peaks = df[df['SampleName'] == sample].copy()
            
            # Determine NoC
            noc = parse_noc_from_sample_name(sample, str(fpath))
            if noc < 1 or noc > MAX_NOC:
                if verbose:
                    print(f"    Skipping sample '{sample}': unknown NoC")
                continue
            
            total_peaks = len(sample_peaks)
            
            # Build tensor
            tensor = build_profile_tensor(sample_peaks, total_peaks)
            
            all_X.append(tensor)
            all_y.append(noc)
            all_names.append(f"{fpath.stem}:{sample}")
    
    if len(all_X) == 0:
        raise ValueError("No profiles loaded! Check data directory and filters.")
    
    X = np.stack(all_X)
    y = np.array(all_y, dtype=np.int64)
    
    if verbose:
        print(f"\nLoaded {len(X)} profiles")
        for noc in sorted(np.unique(y)):
            print(f"  NoC={noc}: {np.sum(y == noc)} profiles")
    
    return X, y, all_names


def train_test_split_alternating(X: np.ndarray, y: np.ndarray,
                                  names: list) -> tuple:
    """
    Split into train/test by taking every other profile (as done in the paper).
    
    Returns: X_train, X_test, y_train, y_test, names_train, names_test
    """
    train_idx = list(range(0, len(X), 2))
    test_idx = list(range(1, len(X), 2))
    
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            [names[i] for i in train_idx],
            [names[i] for i in test_idx])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered"
    
    print("=== Loading PROVEDIt Dataset ===")
    X, y, names = load_provedit_dataset(data_dir)
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"NoC distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Save processed data
    os.makedirs("data/provedit_processed", exist_ok=True)
    np.save("data/provedit_processed/X_gf25.npy", X)
    np.save("data/provedit_processed/y_gf25.npy", y)
    print("Saved to data/provedit_processed/X_gf25.npy and y_gf25.npy")
