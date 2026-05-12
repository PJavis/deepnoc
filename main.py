"""
deepNoC: Deep learning system for NoC assignment in STR DNA profiles.

Replication of Taylor & Humphries (2024).

Usage:
    python main.py prepare   --data-dir <path>          # Process PROVEDIt CSVs → .npy
    python main.py baseline  [--data-dir <path>]        # Run MAC + RF baselines
    python main.py train     [--model full|simple]      # Train deepNoC
    python main.py evaluate  --checkpoint <path>        # Evaluate saved model
    python main.py all                                  # Run everything in sequence
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_prepare(args):
    """Process PROVEDIt CSV files into numpy arrays."""
    from src.data_loader import load_provedit_dataset
    
    print("=" * 60)
    print("  Stage 1: Preparing PROVEDIt data")
    print("=" * 60)
    
    X, y, names = load_provedit_dataset(
        data_dir=args.data_dir,
        kit_filter=args.kit,
        injection_filter=args.injection,
        instrument_filter=args.instrument,
    )
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "X_gf25.npy"), X)
    np.save(os.path.join(out_dir, "y_gf25.npy"), y)
    
    # Save sample names for reference
    with open(os.path.join(out_dir, "sample_names.txt"), 'w') as f:
        for name in names:
            f.write(f"{name}\n")
    
    print(f"\nSaved to {out_dir}:")
    print(f"  X_gf25.npy: {X.shape}")
    print(f"  y_gf25.npy: {y.shape}")
    print(f"  sample_names.txt: {len(names)} samples")


def cmd_baseline(args):
    """Run baseline models (MAC, Random Forest)."""
    from src.data_loader import train_test_split_alternating
    from models.baseline.baselines import run_mac_baseline, train_random_forest
    from src.evaluation import full_evaluation
    
    print("=" * 60)
    print("  Stage 2: Running Baselines")
    print("=" * 60)
    
    # Load data
    X, y = load_data(args)
    
    # Split
    X_train, X_test, y_train, y_test, _, _ = train_test_split_alternating(
        X, y, list(range(len(y)))
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. MAC Baseline
    print("\n--- MAC Baseline ---")
    mac_acc, mac_preds = run_mac_baseline(X_test, y_test)
    labels = sorted(set(y_test))
    full_evaluation(y_test, mac_preds, class_labels=labels,
                    title="MAC", save_dir=args.results_dir)
    
    # 2. Random Forest
    print("\n--- Random Forest Baseline ---")
    rf_model, rf_train_acc, rf_test_acc, rf_preds = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    full_evaluation(y_test, rf_preds, class_labels=labels,
                    title="RandomForest", save_dir=args.results_dir)
    
    print(f"\n{'='*40}")
    print(f"Baseline Summary:")
    print(f"  MAC:           {mac_acc:.4f}")
    print(f"  Random Forest: {rf_test_acc:.4f}")
    print(f"{'='*40}")


def _split(X, y, names, args):
    """Choose the split strategy based on `--split`."""
    strategy = getattr(args, "split", "alternating")
    if strategy == "stratified":
        from src.split import stratified_split
        return stratified_split(X, y, names, test_size=args.test_size, seed=args.seed)
    if strategy == "grouped":
        from src.split import grouped_stratified_split
        return grouped_stratified_split(X, y, names, test_size=args.test_size,
                                        seed=args.seed)
    from src.data_loader import train_test_split_alternating
    return train_test_split_alternating(X, y, names)


def _load_names(args):
    path = os.path.join(args.output_dir, "sample_names.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    return None


def cmd_train(args):
    """Train deepNoC or NoCFormer model."""
    from src.evaluation import full_evaluation, plot_training_history

    print("=" * 60)
    print(f"  Stage 3: Training {args.model}")
    print("=" * 60)

    X, y = load_data(args)
    names = _load_names(args) or [str(i) for i in range(len(y))]
    if len(names) != len(y):
        names = [str(i) for i in range(len(y))]
    X_train, X_test, y_train, y_test, _, _ = _split(X, y, names, args)

    num_classes = int(y.max())
    print(f"Classes: {num_classes} (1 to {num_classes})  "
          f"split={getattr(args, 'split', 'alternating')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == "nocnet_v2":
        from models.nocnet_v2.train import (
            train_nocnet_v2, predict_nocnet_v2, TrainConfig,
        )
        cfg = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, d_model=args.d_model,
            n_heads=args.n_heads, peak_layers=args.peak_layers,
            locus_layers=args.locus_layers, dropout=args.dropout,
            early_stop_patience=args.early_stop_patience,
            p_synth=args.p_synth, samples_per_epoch=args.samples_per_epoch,
        )
        model, history = train_nocnet_v2(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            synth_dir=(None if args.no_synth else args.synth_dir),
            config=cfg, save_dir=args.results_dir,
            tag="nocnet_v2", device=device,
        )
        plot_training_history(
            history, title="NoCNet-v2",
            save_path=os.path.join(args.results_dir, "training_history_nocnet_v2.png"),
        )
        print("\n--- Final Evaluation on Test Set ---")
        probs, preds = predict_nocnet_v2(model, X_test,
                                         batch_size=args.batch_size,
                                         device=device)
        labels = sorted(set(y_test))
        full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                        title="NoCNet-v2", save_dir=args.results_dir)
        return

    if args.model == "nocformer":
        from models.nocformer.train import train_nocformer, predict_with_tta
        model, history = train_nocformer(
            X_train, y_train, X_test, y_test,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            n_heads=args.n_heads,
            peak_layers=args.peak_layers,
            locus_layers=args.locus_layers,
            dropout=args.dropout,
            augment=not args.no_augment,
            early_stop_patience=args.early_stop_patience,
            device=device,
            save_dir=args.results_dir,
            tag="nocformer",
        )
        plot_training_history(
            history, title="NoCFormer",
            save_path=os.path.join(args.results_dir, "training_history_nocformer.png"),
        )
        if args.no_tta:
            print("\n--- Final deterministic evaluation on Test Set ---")
            labels = sorted(set(y_test))
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(device)
                outputs = model(X_test_t)
                probs = outputs['profile_noc_probs'].cpu().numpy()
                preds = probs.argmax(axis=1) + 1
            full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                            title="NoCFormer", save_dir=args.results_dir)
        else:
            print("\n--- Final TTA Evaluation on Test Set ---")
            probs, entropy, preds = predict_with_tta(
                model, X_test, n_samples=args.tta_samples, device=device,
            )
            labels = sorted(set(y_test))
            full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                            title="NoCFormer", save_dir=args.results_dir)
            np.save(os.path.join(args.results_dir, "nocformer_test_entropy.npy"), entropy)
        return

    # Original deepNoC paths
    from models.deepnoc.train import train_deepnoc
    model, history = train_deepnoc(
        X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        device=device,
        save_dir=args.results_dir,
        model_type=args.model,
    )
    plot_training_history(
        history, title=f"deepNoC ({args.model})",
        save_path=os.path.join(args.results_dir, f'training_history_{args.model}.png'),
    )
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        if args.model == "full":
            outputs = model(X_test_t)
            logits = outputs['profile_noc']
        else:
            logits = model(X_test_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1) + 1
    labels = sorted(set(y_test))
    full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                    title=f"deepNoC_{args.model}", save_dir=args.results_dir)


def cmd_synth(args):
    """Generate synthetic mixture pool from NoC=1 PROVEDIt profiles."""
    from src.synth import synthesise

    X = np.load(os.path.join(args.output_dir, "X_gf25.npy"))
    y = np.load(os.path.join(args.output_dir, "y_gf25.npy"))
    pool = X[y == 1].astype(np.float32)
    print(f"[synth] pool size (NoC=1): {pool.shape[0]}")
    Xs, ys, mix, nall = synthesise(
        pool, n_samples=args.n, max_noc=args.max_noc,
        dirichlet_alpha=args.alpha,
        detection_threshold=args.threshold,
        height_jitter_sigma=args.jitter, rng_seed=args.seed,
    )
    os.makedirs(args.synth_dir, exist_ok=True)
    np.save(os.path.join(args.synth_dir, "X.npy"), Xs)
    np.save(os.path.join(args.synth_dir, "y.npy"), ys)
    np.save(os.path.join(args.synth_dir, "mix.npy"), mix)
    np.save(os.path.join(args.synth_dir, "locus_nall.npy"), nall)
    uniq, cnt = np.unique(ys, return_counts=True)
    print(f"[synth] wrote {args.n} profiles to {args.synth_dir}")
    print(f"[synth] NoC distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}")


def cmd_cv(args):
    """Run 5-fold grouped cross-validation across selected models."""
    from src.cv import cross_validate, _load_data
    X, y, names = _load_data(args.output_dir)
    print(f"Loaded X={X.shape} y={y.shape} names={len(names)}")
    cross_validate(
        X, y, names,
        models=args.models,
        n_folds=args.folds,
        seed=args.seed,
        results_root=args.cv_results_dir,
        synth_dir=args.synth_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


def cmd_finetune(args):
    """Fine-tune a NoCNet-v2 synthetic-pretrained checkpoint on real PROVEDIt."""
    from models.nocnet_v2.train import finetune_nocnet_v2
    from src.evaluation import full_evaluation, plot_training_history
    from models.nocnet_v2.train import predict_nocnet_v2

    X, y = load_data(args)
    names = _load_names(args) or [str(i) for i in range(len(y))]
    if len(names) != len(y):
        names = [str(i) for i in range(len(y))]
    X_train, X_test, y_train, y_test, _, _ = _split(X, y, names, args)

    num_classes = int(y.max())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, history = finetune_nocnet_v2(
        args.checkpoint, X_train, y_train, X_test, y_test,
        num_classes=num_classes,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        samples_per_epoch=args.samples_per_epoch,
        swa_frac=args.swa_frac,
        jitter_sigma=args.jitter_sigma,
        dropout_p=args.dropout_p,
        save_dir=args.results_dir,
        tag=args.tag,
        freeze_peak_stages=args.freeze_peak,
        device=device,
    )
    plot_training_history(
        history, title="NoCNet-v2 finetune",
        save_path=os.path.join(args.results_dir,
                               f"training_history_{args.tag}.png"),
    )
    probs, preds = predict_nocnet_v2(model, X_test,
                                     batch_size=args.batch_size,
                                     device=device)
    labels = sorted(set(y_test))
    full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                    title=f"NoCNet-v2_{args.tag}", save_dir=args.results_dir)


def cmd_evaluate(args):
    """Evaluate a saved model checkpoint."""
    from models.deepnoc.train import load_model
    from src.data_loader import train_test_split_alternating
    from src.evaluation import full_evaluation
    
    print("=" * 60)
    print("  Evaluating model checkpoint")
    print("=" * 60)
    
    X, y = load_data(args)
    _, X_test, _, y_test, _, _ = train_test_split_alternating(
        X, y, list(range(len(y)))
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(y.max())
    
    model = load_model(args.checkpoint, device, num_classes, args.model)
    
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        if args.model == "full":
            outputs = model(X_test_t)
            logits = outputs['profile_noc']
        else:
            logits = model(X_test_t)
        
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1) + 1
    
    labels = sorted(set(y_test))
    full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                    title="deepNoC_eval", save_dir=args.results_dir)


def cmd_all(args):
    """Run the full pipeline."""
    print("Running full pipeline...")
    
    # Check if processed data exists
    x_path = os.path.join(args.output_dir, "X_gf25.npy")
    if not os.path.exists(x_path):
        print("\nStep 1/3: Preparing data...")
        cmd_prepare(args)
    else:
        print(f"\nStep 1/3: Data already exists at {x_path}, skipping preparation.")
    
    print("\nStep 2/3: Running baselines...")
    cmd_baseline(args)
    
    print("\nStep 3/3: Training deepNoC...")
    cmd_train(args)


def load_data(args):
    """Load numpy data arrays."""
    x_path = os.path.join(args.output_dir, "X_gf25.npy")
    y_path = os.path.join(args.output_dir, "y_gf25.npy")
    
    if not os.path.exists(x_path):
        print(f"ERROR: Data not found at {x_path}")
        print(f"Run 'python main.py prepare' first, or check --output-dir")
        sys.exit(1)
    
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"NoC distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="deepNoC: Deep learning NoC assignment for STR DNA profiles"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--output-dir', default='data/provedit_processed',
                        help='Directory with processed .npy files')
    common.add_argument('--results-dir', default='results',
                        help='Directory for results and plots')
    
    # Prepare command
    p_prepare = subparsers.add_parser('prepare', parents=[common],
                                       help='Process PROVEDIt CSVs')
    p_prepare.add_argument('--data-dir',
                           default='data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered',
                           help='Path to PROVEDIt CSV directory')
    p_prepare.add_argument('--kit', default='GF', help='Kit filter (GF=GlobalFiler)')
    p_prepare.add_argument('--injection', default='25sec', help='Injection time filter')
    p_prepare.add_argument('--instrument', default='3500', help='Instrument filter')
    
    # Baseline command
    p_baseline = subparsers.add_parser('baseline', parents=[common],
                                        help='Run baselines')
    
    # Train command
    p_train = subparsers.add_parser('train', parents=[common],
                                     help='Train deepNoC or NoCFormer')
    p_train.add_argument('--model',
                         choices=['full', 'simple', 'nocformer', 'nocnet_v2'],
                         default='nocnet_v2',
                         help='Model: full / simple (deepNoC), nocformer, nocnet_v2')
    p_train.add_argument('--epochs', type=int, default=200,
                         help='Number of training epochs')
    p_train.add_argument('--batch-size', type=int, default=32,
                         help='Batch size')
    p_train.add_argument('--lr', type=float, default=3e-4,
                         help='Learning rate')
    p_train.add_argument('--beta1', type=float, default=0.5,
                         help='Adam beta1 parameter (deepNoC only)')
    p_train.add_argument('--split', choices=['alternating', 'stratified', 'grouped'],
                         default='grouped',
                         help='Train/test split strategy')
    p_train.add_argument('--test-size', type=float, default=0.25)
    p_train.add_argument('--seed', type=int, default=42)
    # NoCFormer-only knobs
    p_train.add_argument('--d-model', type=int, default=128)
    p_train.add_argument('--n-heads', type=int, default=4)
    p_train.add_argument('--peak-layers', type=int, default=2)
    p_train.add_argument('--locus-layers', type=int, default=4)
    p_train.add_argument('--dropout', type=float, default=0.15)
    p_train.add_argument('--weight-decay', type=float, default=5e-4,
                         help='Weight decay for NoCFormer AdamW')
    p_train.add_argument('--early-stop-patience', type=int, default=20,
                         help='Stop training if test does not improve for this many epochs')
    p_train.add_argument('--no-augment', action='store_true',
                         help='Disable synthetic-mixture augmentation')
    p_train.add_argument('--no-tta', action='store_true',
                         help='Disable TTA evaluation and use deterministic predictions')
    p_train.add_argument('--tta-samples', type=int, default=20,
                         help='MC-Dropout / TTA samples at evaluation time')
    # NoCNet-v2-only knobs
    p_train.add_argument('--synth-dir', default='data/synthetic',
                         help='Path to synthetic-pool .npy files (NoCNet-v2)')
    p_train.add_argument('--no-synth', action='store_true',
                         help='Disable synthetic data even if present')
    p_train.add_argument('--p-synth', type=float, default=0.8,
                         help='Fraction of batch drawn from synthetic pool')
    p_train.add_argument('--samples-per-epoch', type=int, default=4000,
                         help='Hybrid loader iterations per epoch')
    
    # Synth command
    p_synth = subparsers.add_parser('synth', parents=[common],
                                    help='Generate synthetic mixture pool')
    p_synth.add_argument('--synth-dir', default='data/synthetic')
    p_synth.add_argument('--n', type=int, default=20_000)
    p_synth.add_argument('--max-noc', type=int, default=5)
    p_synth.add_argument('--alpha', type=float, default=1.5)
    p_synth.add_argument('--threshold', type=float, default=50.0)
    p_synth.add_argument('--jitter', type=float, default=0.08)
    p_synth.add_argument('--seed', type=int, default=0)

    # CV command
    p_cv = subparsers.add_parser('cv', parents=[common],
                                 help='5-fold grouped cross-validation')
    p_cv.add_argument('--cv-results-dir', default='results/cv')
    p_cv.add_argument('--synth-dir', default='data/synthetic')
    p_cv.add_argument('--models', nargs='+',
                      default=['mac', 'rf', 'nocnet_v2'],
                      choices=['mac', 'rf', 'deepnoc_simple', 'deepnoc_full',
                               'nocformer', 'nocnet_v2'])
    p_cv.add_argument('--folds', type=int, default=5)
    p_cv.add_argument('--epochs', type=int, default=60)
    p_cv.add_argument('--batch-size', type=int, default=16)
    p_cv.add_argument('--lr', type=float, default=3e-4)
    p_cv.add_argument('--seed', type=int, default=42)

    # Finetune command
    p_ft = subparsers.add_parser('finetune', parents=[common],
                                 help='Fine-tune NoCNet-v2 on real PROVEDIt')
    p_ft.add_argument('--checkpoint', required=True,
                      help='Synth-pretrained NoCNet-v2 ckpt (best_nocnet_v2.pt)')
    p_ft.add_argument('--epochs', type=int, default=30)
    p_ft.add_argument('--lr', type=float, default=1e-5)
    p_ft.add_argument('--batch-size', type=int, default=16)
    p_ft.add_argument('--weight-decay', type=float, default=1e-4)
    p_ft.add_argument('--samples-per-epoch', type=int, default=1500)
    p_ft.add_argument('--swa-frac', type=float, default=0.4)
    p_ft.add_argument('--jitter-sigma', type=float, default=0.05)
    p_ft.add_argument('--dropout-p', type=float, default=0.01)
    p_ft.add_argument('--freeze-peak', action='store_true',
                      help='Freeze peak stages, only fine-tune cross-locus + heads')
    p_ft.add_argument('--tag', default='nocnet_v2_ft')
    p_ft.add_argument('--split', choices=['alternating', 'stratified', 'grouped'],
                      default='grouped')
    p_ft.add_argument('--test-size', type=float, default=0.25)
    p_ft.add_argument('--seed', type=int, default=42)

    # Evaluate command
    p_eval = subparsers.add_parser('evaluate', parents=[common],
                                    help='Evaluate checkpoint')
    p_eval.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    p_eval.add_argument('--model', choices=['full', 'simple'], default='simple')
    
    # All command
    p_all = subparsers.add_parser('all', parents=[common],
                                   help='Run full pipeline')
    p_all.add_argument('--data-dir',
                       default='data/provedit_processed/PROVEDIt_1-5-Person CSVs Filtered',
                       help='Path to PROVEDIt CSV directory')
    p_all.add_argument('--kit', default='GF')
    p_all.add_argument('--injection', default='25sec')
    p_all.add_argument('--instrument', default='3500')
    p_all.add_argument('--model', choices=['full', 'simple'], default='simple')
    p_all.add_argument('--epochs', type=int, default=2000)
    p_all.add_argument('--batch-size', type=int, default=100)
    p_all.add_argument('--lr', type=float, default=1e-5)
    p_all.add_argument('--beta1', type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    commands = {
        'prepare': cmd_prepare,
        'baseline': cmd_baseline,
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'all': cmd_all,
        'synth': cmd_synth,
        'cv': cmd_cv,
        'finetune': cmd_finetune,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()