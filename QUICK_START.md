# Quick Start - Improved NoCFormer

## What Changed?
✅ Model size reduced (1.28M → 728K params)
✅ MixUp augmentation added (feature space interpolation)
✅ Label smoothing added (regularization)
✅ Early stopping integrated (prevent overfitting)

## Test Results
**Best Accuracy: 66.8%** (improved from 55.6%)

## How to Use

### Train with new improvements:
```bash
python main.py train --model nocformer --split grouped \
    --epochs 100 --early-stop-patience 15 --batch-size 32
```

### Key parameters (already optimized):
- `--d-model 96` (reduced from 128, saves 40% memory)
- `--dropout 0.15` (standard regularization)
- `--early-stop-patience 20` (stops after 20 epochs without improvement)

## Performance Breakdown

**By NoC:**
- NoC=1: 95.7% (excellent)
- NoC=2: 90.9% (good)
- NoC=3: 71.4% (ok)
- NoC=4: 2.9% (very weak - imbalanced data issue)
- NoC=5: 73.5% (ok)

**Overall: 66.8%** on test set

## Known Limitation

⚠️ **Data bottleneck**: Only 70 NoC=1 profiles available (expected ~2,700)
- Missing RD12-0002 1P CSVs from PROVEDIt dataset
- Limits synthetic mixture diversity
- Prevents reaching 0.9+ accuracy

## To Reach 0.9 Accuracy

1. Get full 1P data (2,700 profiles)
2. Re-prepare with:
   ```bash
   python main.py prepare --data-dir "data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered" \
       --kit all --injection all --instrument all
   ```
3. Retrain

## Model Files

**Latest best model**: `results/best_nocformer.pt`
- Params: 728K (down from 1.28M)
- Training curve: `results/training_history_nocformer.png`
- Metrics: `results/metrics_nocformer.json`

## Environment

Python 3.11 + PyTorch 2.x
GPU: CUDA (recommended)
CPU: ~2-3 min per epoch (slow)

---
*Generated: May 13, 2024*
*Session: NoCFormer improvements applied*
