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


def cmd_train(args):
    """Train deepNoC model."""
    from src.data_loader import train_test_split_alternating
    from models.deepnoc.train import train_deepnoc
    from src.evaluation import full_evaluation, plot_training_history
    
    print("=" * 60)
    print(f"  Stage 3: Training deepNoC ({args.model})")
    print("=" * 60)
    
    # Load data
    X, y = load_data(args)
    
    # Split
    X_train, X_test, y_train, y_test, _, _ = train_test_split_alternating(
        X, y, list(range(len(y)))
    )
    
    # Determine number of classes
    num_classes = int(y.max())
    print(f"Classes: {num_classes} (1 to {num_classes})")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
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
    
    # Plot training history
    plot_training_history(
        history, title=f"deepNoC ({args.model})",
        save_path=os.path.join(args.results_dir, f'training_history_{args.model}.png'),
    )
    
    # Final evaluation
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
        preds = probs.argmax(axis=1) + 1  # 1-indexed
    
    labels = sorted(set(y_test))
    full_evaluation(y_test, preds, y_probs=probs, class_labels=labels,
                    title=f"deepNoC_{args.model}", save_dir=args.results_dir)


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
                                     help='Train deepNoC')
    p_train.add_argument('--model', choices=['full', 'simple'], default='simple',
                         help='Model type: full (multi-output) or simple')
    p_train.add_argument('--epochs', type=int, default=2000,
                         help='Number of training epochs')
    p_train.add_argument('--batch-size', type=int, default=100,
                         help='Batch size')
    p_train.add_argument('--lr', type=float, default=1e-5,
                         help='Learning rate')
    p_train.add_argument('--beta1', type=float, default=0.5,
                         help='Adam beta1 parameter')
    
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
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()