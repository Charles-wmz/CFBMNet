"""
5-Fold Cross-Validation Training and Testing Script.
"""
import os
import logging
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import json
from collections import defaultdict
from datetime import datetime

# Set environment variables for reproducibility BEFORE importing other modules
# CRITICAL: These must be set before importing torch
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # For better memory management

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from config import Config
from utils.dataset import RespiratoryFlowDataset
from models.model_registry import (
    create_torch_model,
    count_torch_parameters,
    count_trainable_torch_parameters,
)
from trainer import RegressionTrainer
from utils.metrics import MetricsCalculator
import pandas as pd

# Set up logging.
# We attach a FileHandler later after we know output_dir (cv_results/{exp}).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Set random seed for complete reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # Set CUDA random seeds
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Force deterministic behavior in CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable deterministic algorithms with warn_only=True to allow some operations
        # that don't have deterministic implementations (like upsample)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Fallback for older PyTorch versions that don't support warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except:
                pass
        
        # Disable TF32 for better reproducibility
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def create_kfold_splits(sample_ids, n_folds=5, random_seed=42):
    """
    Create K-fold splits ensuring subject-independent splitting.
    
    Args:
        sample_ids: List of sample IDs
        n_folds: Number of folds
        random_seed: Random seed
    
    Returns:
        List of (train_ids, test_ids) tuples for each fold
    """
    # Group samples by subject
    subject_samples = defaultdict(list)
    for sid in sample_ids:
        subject_id = sid.split('_')[0]
        subject_samples[subject_id].append(sid)
    
    # Get unique subjects and shuffle
    subjects = list(subject_samples.keys())
    rng = np.random.RandomState(random_seed)
    rng.shuffle(subjects)
    
    # Split subjects into folds
    fold_subjects = np.array_split(subjects, n_folds)
    
    # Create train/test splits
    folds = []
    for i in range(n_folds):
        test_subjects = set(fold_subjects[i])
        train_subjects = set(subjects) - test_subjects
        
        train_ids = []
        test_ids = []
        
        # Sort subjects to ensure deterministic order
        for subject in sorted(train_subjects):
            # Sort samples within each subject for deterministic order
            train_ids.extend(sorted(subject_samples[subject]))
        for subject in sorted(test_subjects):
            # Sort samples within each subject for deterministic order
            test_ids.extend(sorted(subject_samples[subject]))
        
        folds.append((train_ids, test_ids))
        
        logger.info(f"Fold {i+1}: {len(train_subjects)} train subjects ({len(train_ids)} samples), "
                   f"{len(test_subjects)} test subjects ({len(test_ids)} samples)")
    
    return folds


def log_training_summary(valid_samples, labels_df, config=None):
    """
    Log subject counts, sample counts, sex distribution, and model parameters.
    """
    if config is None:
        config = Config
    
    # Subjects and samples
    subject_ids = set(sid.split('_')[0] for sid in valid_samples)
    n_subjects = len(subject_ids)
    n_samples = len(valid_samples)

    # Sex distribution by subject id
    n_male = None
    n_female = None
    try:
        if labels_df is not None and ('id' in labels_df.columns) and ('gender' in labels_df.columns):
            labels_sub = labels_df[labels_df['id'].astype(int).astype(str).str.zfill(4).isin(subject_ids)]
            # gender: 1=male, 0=female (per data_pre/label.csv)
            n_male = int((labels_sub['gender'] == 1).sum())
            n_female = int((labels_sub['gender'] == 0).sum())
    except Exception as e:
        logger.warning(f"Sex distribution summary failed: {e}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("Training Data and Model Summary")
    logger.info("="*70)
    logger.info(f"Number of subjects: {n_subjects}")
    logger.info(f"Number of samples:  {n_samples}")
    if (n_male is not None) and (n_female is not None):
        logger.info(f"Sex distribution:   male={n_male}, female={n_female}")
    logger.info("-"*70)
    
    # Model architecture parameters from config
    logger.info("Model architecture:")
    logger.info(f"  MODEL = {getattr(config, 'MODEL', 'direct')}")
    logger.info(f"  CONDITION_DIM = {config.CONDITION_DIM}")
    logger.info(f"  CONDITION_ENCODER_HIDDEN_DIMS = {config.CONDITION_ENCODER_HIDDEN_DIMS}")
    logger.info(f"  Innovation 1 (basis-mixture decoder) = {bool(getattr(config, 'USE_SPECTRO_TEMPORAL_CONDITION_ENCODER', False))}")
    _use_demo = bool(getattr(config, 'USE_DEMOGRAPHIC', False))
    logger.info(f"  Innovation 2 (demographic encoder) = {_use_demo}")
    if _use_demo:
        logger.info(f"    DEMOGRAPHIC_FEATURES = {getattr(config, 'DEMOGRAPHIC_FEATURES', [])}")
    logger.info(f"  SpecAugment = {bool(getattr(config, 'SPEC_AUGMENT', False))}")
    logger.info("-"*70)
    use_phys_smooth = bool(getattr(config, 'USE_PHYS_SMOOTH_LOSS', False))
    _uw = getattr(config, 'UW_WARMUP_EPOCHS', 50)
    uw_warmup = 50 if _uw is None else int(_uw)
    logger.info("Loss configuration:")
    logger.info(f"  LOSS_TYPE = {getattr(config, 'LOSS_TYPE', 'l1')}")
    logger.info(f"  LAMBDA_REG = {getattr(config, 'LAMBDA_REG', 1.0)}")
    logger.info(f"  Innovation 3 (physics + smoothness + UW) = {use_phys_smooth}")
    if use_phys_smooth:
        logger.info(f"    UW_WARMUP_EPOCHS = {uw_warmup}")
        logger.info(f"    LAMBDA_PHYS = {getattr(config, 'LAMBDA_PHYS', 1.0)}, LAMBDA_SMOOTH = {getattr(config, 'LAMBDA_SMOOTH', 0.1)}")
        logger.info(f"    FEV1_WEIGHT = {getattr(config, 'FEV1_WEIGHT', 1.0)}, FVC_WEIGHT = {getattr(config, 'FVC_WEIGHT', 1.0)}, PEF_WEIGHT = {getattr(config, 'PEF_WEIGHT', 1.0)}")
    
    # Model parameter count
    try:
        model = create_torch_model(config)
        total_params = count_torch_parameters(model)
        trainable_params = count_trainable_torch_parameters(model)
        logger.info(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")
    except Exception as e:
        logger.warning(f"Parameter counting failed: {e}")
    
    logger.info("="*70)
    logger.info("")

def train_fold(fold_idx, train_ids, val_ids, output_dir, config=None):
    """
    Train a single fold.
    
    Args:
        fold_idx: Fold index
        train_ids: Training sample IDs
        val_ids: Validation sample IDs
        output_dir: Output directory
        config: Configuration object (if None, uses global Config)
    """
    if config is None:
        config = Config

    logger.info(f"\n{'='*70}")
    logger.info(f"Training Fold {fold_idx + 1} (CFBMNet)")
    logger.info(f"{'='*70}")
    
    # Set seed for this fold to ensure reproducibility
    # Use a different seed for each fold based on the base seed
    fold_seed = config.RANDOM_SEED + fold_idx
    set_seed(fold_seed)
    logger.info(f"Set random seed to {fold_seed} for Fold {fold_idx + 1}")
    
    # Create datasets
    # IMPORTANT: Sort IDs to ensure deterministic order
    train_ids_sorted = sorted(train_ids)
    val_ids_sorted = sorted(val_ids)
    demographic_features = getattr(config, 'DEMOGRAPHIC_FEATURES', ['gender', 'age', 'height', 'weight']) if bool(getattr(config, 'USE_DEMOGRAPHIC', False)) else []
    train_dataset = RespiratoryFlowDataset(
        mel_dir=config.MEL_DIR,
        csv_dir=config.CSV_DIR,
        label_file=config.LABEL_FILE,
        sample_ids=train_ids_sorted,
        demographic_features=demographic_features,
        training=True,
        spec_augment=bool(getattr(config, 'SPEC_AUGMENT', False)),
    )
    
    val_dataset = RespiratoryFlowDataset(
        mel_dir=config.MEL_DIR,
        csv_dir=config.CSV_DIR,
        label_file=config.LABEL_FILE,
        sample_ids=val_ids_sorted,
        demographic_features=demographic_features,
    )
    
    # Worker init function for DataLoader reproducibility
    def worker_init_fn(worker_id):
        np.random.seed(fold_seed + worker_id)
        random.seed(fold_seed + worker_id)
        torch.manual_seed(fold_seed + worker_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold_seed + worker_id)
            torch.cuda.manual_seed_all(fold_seed + worker_id)
    
    # Create data loaders with fixed generator for reproducibility
    # Generator must be on CPU for DataLoader
    train_generator = torch.Generator()
    train_generator.manual_seed(fold_seed)
    
    # Create generator for validation loader (even with shuffle=False, generator ensures deterministic order)
    val_generator = torch.Generator()
    val_generator.manual_seed(fold_seed + 10000)  # Different seed from train to avoid overlap
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        generator=train_generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        generator=val_generator  # Ensure deterministic order
    )
    
    # Create model - ensure deterministic initialization by setting seed again
    set_seed(fold_seed)
    model = create_torch_model(config)
    model = model.to(config.DEVICE)
    
    # Checkpoint directory (no fold subfolders)
    pth_dir = os.path.join(output_dir, 'pth')
    os.makedirs(pth_dir, exist_ok=True)
    lambda_reg = getattr(config, 'LAMBDA_REG', 1.0)
    use_phys_smooth = bool(getattr(config, 'USE_PHYS_SMOOTH_LOSS', False))
    if use_phys_smooth:
        lambda_phys = getattr(config, 'LAMBDA_PHYS', 1.0)
        lambda_smooth = getattr(config, 'LAMBDA_SMOOTH', 0.1)
    else:
        lambda_phys = 0.0
        lambda_smooth = 0.0
    sob_alpha = float(getattr(config, 'SOB_ALPHA', 0.2))
    sob_beta = float(getattr(config, 'SOB_BETA', 0.05))
    sob_delta = float(getattr(config, 'SOB_DELTA', 1.0))
    fev1_weight = getattr(config, 'FEV1_WEIGHT', 1.0)
    fvc_weight = getattr(config, 'FVC_WEIGHT', 1.0)
    pef_weight = getattr(config, 'PEF_WEIGHT', 2.0)

    # Innovation 3 internal: uncertainty weighting for enabled loss terms
    from trainer import UncertaintyLossWeighter
    loss_weighter = None
    if (lambda_phys > 0) or (lambda_smooth > 0):
        keys = ['reg']
        if lambda_phys > 0:
            keys.append('phys')
        if lambda_smooth > 0:
            keys.append('smooth')
        loss_weighter = UncertaintyLossWeighter(keys).to(config.DEVICE)

    # Create optimizer (include loss_weighter params if enabled)
    opt_params = list(model.parameters())
    if loss_weighter is not None:
        opt_params = opt_params + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Cosine annealing with warm restarts: T_0 = total epochs (single cosine cycle).
    # Periodically recovers LR, helping escape local minima on small datasets.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.NUM_EPOCHS,
        T_mult=1,
        eta_min=1e-6,
    )
    
    trainer = RegressionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_reg=lambda_reg,
        lambda_phys=lambda_phys,
        loss_type=config.LOSS_TYPE,
        lambda_smooth=lambda_smooth,
        sob_alpha=sob_alpha,
        sob_beta=sob_beta,
        sob_delta=sob_delta,
        fev1_weight=fev1_weight,
        fvc_weight=fvc_weight,
        pef_weight=pef_weight,
        loss_weighter=loss_weighter,
        # allow 0 warmup (disable) without being overridden by "or 0" patterns
        uw_warmup_epochs=0 if getattr(config, 'UW_WARMUP_EPOCHS', 0) is None else int(getattr(config, 'UW_WARMUP_EPOCHS', 0)),
        grad_clip_norm=float(getattr(config, 'GRAD_CLIP_NORM', 1.0)),
        device=config.DEVICE,
        checkpoint_dir=pth_dir,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        fold_info=f'Fold {fold_idx+1}/5',
        checkpoint_tag=f'fold{fold_idx+1}'
    )
    
    # Train
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    # Plot and save loss curves
    loss_plot_path = os.path.join(pth_dir, f'loss_curves_fold{fold_idx+1}.png')
    trainer.plot_loss_curves(save_path=loss_plot_path)
    logger.info(f"Loss curves saved to {loss_plot_path}")
    
    return os.path.join(pth_dir, f'best_model_fold{fold_idx+1}.pth')


def test_fold(fold_idx, test_ids, checkpoint_path, output_dir, config=None):
    """
    Test a single fold.
    
    Args:
        fold_idx: Fold index
        test_ids: Test sample IDs
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory
        config: Configuration object (if None, uses global Config)
    """
    if config is None:
        config = Config
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing Fold {fold_idx + 1}")
    logger.info(f"{'='*70}")
    
    # Set seed for reproducible testing
    fold_seed = config.RANDOM_SEED + fold_idx
    set_seed(fold_seed)
    
    # Create test dataset
    demographic_features = getattr(config, 'DEMOGRAPHIC_FEATURES', ['gender', 'age', 'height', 'weight']) if bool(getattr(config, 'USE_DEMOGRAPHIC', False)) else []
    test_dataset = RespiratoryFlowDataset(
        mel_dir=config.MEL_DIR,
        csv_dir=config.CSV_DIR,
        label_file=config.LABEL_FILE,
        sample_ids=test_ids,
        demographic_features=demographic_features
    )
    
    # Worker init function for DataLoader reproducibility
    def worker_init_fn(worker_id):
        np.random.seed(fold_seed + worker_id)
        random.seed(fold_seed + worker_id)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn
    )
    
    import time

    set_seed(fold_seed)
    model = create_torch_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Warmup forward pass to exclude JIT/compilation overhead from timing
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        try:
            warmup_batch = next(iter(test_loader))
            warmup_mel = warmup_batch['mel'].to(config.DEVICE)
            warmup_demo = warmup_batch.get('demographic', None)
            if warmup_demo is not None:
                warmup_demo = warmup_demo.to(config.DEVICE)
            with torch.no_grad():
                if warmup_demo is not None:
                    model(warmup_mel, warmup_demo)
                else:
                    model(warmup_mel)
                torch.cuda.synchronize()
        except StopIteration:
            pass

    all_predictions = []
    all_ground_truths = []
    all_labels = []
    all_sample_ids = []
    inference_seconds_total = 0.0
    inference_samples_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            mel = batch['mel']
            flow_gt = batch['flow'].cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            sample_ids = batch['sample_id']
            demographic = batch.get('demographic', None)

            mel = mel.to(config.DEVICE)
            if demographic is not None:
                demographic = demographic.to(config.DEVICE)
            if config.DEVICE == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if demographic is not None:
                flow_pred = model(mel, demographic)
            else:
                flow_pred = model(mel)
            flow_pred = torch.clamp(flow_pred, min=0.0)
            if config.DEVICE == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            flow_pred = flow_pred.cpu().numpy()
            
            inference_seconds_total += (t1 - t0)
            inference_samples_total += int(batch['mel'].shape[0])
            
            all_predictions.append(flow_pred)
            all_ground_truths.append(flow_gt)
            all_labels.append(labels)
            all_sample_ids.extend(sample_ids)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    metrics = MetricsCalculator.compute_all_metrics(
        all_predictions,
        all_ground_truths,
        all_labels
    )
    
    # Single-sample inference time (ms/sample), measured around solver.sample(...)
    if inference_samples_total > 0:
        inference_ms_per_sample = (inference_seconds_total * 1000.0) / float(inference_samples_total)
    else:
        inference_ms_per_sample = float('nan')
    metrics['inference_ms_per_sample'] = float(inference_ms_per_sample)
    logger.info(
        f"Fold {fold_idx + 1} Inference Time: {inference_ms_per_sample:.3f} ms/sample "
        f"(total {inference_seconds_total:.3f}s over {inference_samples_total} samples)"
    )
    
    logger.info(f"Fold {fold_idx + 1} Test Metrics:")
    MetricsCalculator.print_metrics(metrics, prefix=f'Fold {fold_idx + 1}')
    
    # Note: sample-curve plots are disabled to avoid writing exp/fold/plots outputs.
    
    # Return metrics and sample-level data
    sample_data = {
        'fold_idx': fold_idx,
        'sample_ids': all_sample_ids,
        'flow_pred': all_predictions,
        'flow_gt': all_ground_truths,
        'labels_gt': all_labels  # (N, 3) [FEV1, FVC, PEF]
    }
    
    return metrics, sample_data


def generate_sample_results(all_sample_data, output_dir, config=None):
    """
    Generate CSV file with sample-level metrics (results_sample.csv).
    
    Args:
        all_sample_data: List of dictionaries, each containing sample data from one fold
        output_dir: Output directory
        config: Configuration object
    """
    if config is None:
        config = Config
    
    # Collect all samples from all folds
    all_samples = []
    
    for fold_data in all_sample_data:
        fold_idx = int(fold_data.get('fold_idx', 0))
        fold_label = fold_idx + 1  # 1..K for CSV / plotting
        sample_ids = fold_data['sample_ids']
        flow_pred = fold_data['flow_pred']  # (N, 60)
        flow_gt = fold_data['flow_gt']  # (N, 60)
        labels_gt = fold_data['labels_gt']  # (N, 3) [FEV1, FVC, PEF]

        # Time step between adjacent points (inclusive endpoints in preprocess).
        seq_len = int(flow_pred.shape[1])
        dt = 3.0 / max(seq_len - 1, 1)
        
        for i, sample_id in enumerate(sample_ids):
            # Flow metrics
            flow_pred_i = flow_pred[i]  # (60,)
            flow_gt_i = flow_gt[i]  # (60,)
            
            flow_mae = np.mean(np.abs(flow_pred_i - flow_gt_i))
            flow_rmse = np.sqrt(np.mean((flow_pred_i - flow_gt_i) ** 2))
            # MAPE for flow (avoid division by zero)
            threshold = 0.1
            mask = np.abs(flow_gt_i) > threshold
            if np.sum(mask) > 0:
                flow_mape = np.mean(np.abs((flow_pred_i[mask] - flow_gt_i[mask]) / flow_gt_i[mask])) * 100
            else:
                flow_mape = 0.0
            
            # FVC: integral from 0 to 3s (all 60 points)
            fvc_pred = np.trapz(flow_pred_i, dx=dt)
            fvc_gt = labels_gt[i, 1]
            
            fvc_mae = np.abs(fvc_pred - fvc_gt)
            fvc_rmse = np.sqrt((fvc_pred - fvc_gt) ** 2)
            fvc_mape = np.abs((fvc_pred - fvc_gt) / (fvc_gt + 1e-8)) * 100
            
            # FEV1: integral from 0 to exactly 1s (with linear interpolation at boundary)
            idx_1s = int(1.0 / dt)
            t_left = idx_1s * dt
            alpha_1s = (1.0 - t_left) / dt
            f_at_1s = flow_pred_i[idx_1s] * (1.0 - alpha_1s) + flow_pred_i[idx_1s + 1] * alpha_1s
            fev1_pred = (np.trapz(flow_pred_i[:idx_1s + 1], dx=dt)
                         + 0.5 * (flow_pred_i[idx_1s] + f_at_1s) * (1.0 - t_left))
            fev1_gt = labels_gt[i, 0]
            
            fev1_mae = np.abs(fev1_pred - fev1_gt)
            fev1_rmse = np.sqrt((fev1_pred - fev1_gt) ** 2)
            fev1_mape = np.abs((fev1_pred - fev1_gt) / (fev1_gt + 1e-8)) * 100
            
            # PEF: maximum flow
            pef_pred = np.max(flow_pred_i)
            pef_gt = labels_gt[i, 2]

            pef_mae = np.abs(pef_pred - pef_gt)
            pef_rmse = np.sqrt((pef_pred - pef_gt) ** 2)
            pef_mape = np.abs((pef_pred - pef_gt) / (pef_gt + 1e-8)) * 100

            # FEV1/FVC ratio
            fev1_fvc_pred = fev1_pred / (fvc_pred + 1e-8)
            fev1_fvc_gt = fev1_gt / (fvc_gt + 1e-8)
            fev1_fvc_mae = np.abs(fev1_fvc_pred - fev1_fvc_gt)
            fev1_fvc_rmse = np.sqrt((fev1_fvc_pred - fev1_fvc_gt) ** 2)
            fev1_fvc_mape = np.abs((fev1_fvc_pred - fev1_fvc_gt) / (fev1_fvc_gt + 1e-8)) * 100

            # Average MAPE of FVC and FEV1
            avg_fvc_fev1_mape = (fvc_mape + fev1_mape) / 2.0

            all_samples.append({
                'sample_id': sample_id,
                'fold': fold_label,
                'flow_mae': flow_mae,
                'flow_rmse': flow_rmse,
                'flow_mape': flow_mape,
                'fvc_mae': fvc_mae,
                'fvc_rmse': fvc_rmse,
                'fvc_mape': fvc_mape,
                'fev1_mae': fev1_mae,
                'fev1_rmse': fev1_rmse,
                'fev1_mape': fev1_mape,
                'pef_mae': pef_mae,
                'pef_rmse': pef_rmse,
                'pef_mape': pef_mape,
                'fev1_fvc_mae': fev1_fvc_mae,
                'fev1_fvc_rmse': fev1_fvc_rmse,
                'fev1_fvc_mape': fev1_fvc_mape,
                'avg_fvc_fev1_mape': avg_fvc_fev1_mape
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Reorder columns
    column_order = [
        'sample_id',
        'fold',
        'flow_mae', 'flow_rmse', 'flow_mape',
        'fvc_mae', 'fvc_rmse', 'fvc_mape',
        'fev1_mae', 'fev1_rmse', 'fev1_mape',
        'pef_mae', 'pef_rmse', 'pef_mape',
        'fev1_fvc_mae', 'fev1_fvc_rmse', 'fev1_fvc_mape',
        'avg_fvc_fev1_mape'
    ]
    df = df[column_order]
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'results_sample.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Sample-level results saved to {csv_path}")
    logger.info(f"Total samples: {len(df)}")


def generate_sample_point_matrix_csv(all_sample_data, output_dir, config=None):
    """
    Generate a wide point-level CSV:
    - 4 rows per sample
      1) <sample_id>_flow_true
      2) <sample_id>_flow_pred
      3) <sample_id>_volume_true
      4) <sample_id>_volume_pred
    - each row contains point-wise values for all time points (t00-t59)
    """
    if config is None:
        config = Config

    # Collect sample-level curves across folds
    sample_to_pred = {}
    sample_to_gt = {}
    sample_to_fvc_label = {}
    seq_len = None

    for fold_data in all_sample_data:
        sample_ids = fold_data['sample_ids']
        flow_pred = fold_data['flow_pred']  # (N, T)
        flow_gt = fold_data['flow_gt']      # (N, T)
        labels_gt = fold_data.get('labels_gt', None)  # (N, 3): [FEV1, FVC, PEF]

        if seq_len is None:
            seq_len = int(flow_pred.shape[1])

        for i, sid in enumerate(sample_ids):
            sid = str(sid)
            # one sample should appear only once in 5-fold test union
            if sid in sample_to_pred:
                continue
            sample_to_pred[sid] = np.asarray(flow_pred[i], dtype=np.float64)
            sample_to_gt[sid] = np.asarray(flow_gt[i], dtype=np.float64)
            if labels_gt is not None:
                sample_to_fvc_label[sid] = float(labels_gt[i, 1])

    if seq_len is None or len(sample_to_pred) == 0:
        logger.warning("No sample point data found. Skip matrix csv generation.")
        return

    # Uniform reference grid (fallback)
    dt = 3.0 / max(seq_len - 1, 1)
    uniform_time = np.linspace(0.0, 3.0, seq_len, dtype=np.float64)
    point_cols = [f"t{i:02d}" for i in range(seq_len)]

    # Unit sanity check:
    # compare integrated true flow (L/s * s -> L) with label FVC (should also be in L).
    # If ratio is around 1000 or 0.001, it strongly suggests unit mismatch.
    fvc_ratios = []
    for sid in sorted(sample_to_gt.keys()):
        if sid not in sample_to_fvc_label:
            continue
        fvc_label = sample_to_fvc_label[sid]
        if fvc_label <= 1e-8:
            continue
        flow_curve = sample_to_gt[sid]
        fvc_from_flow = float(np.trapz(flow_curve, dx=dt))
        fvc_ratios.append(fvc_from_flow / fvc_label)

    if len(fvc_ratios) > 0:
        ratio_arr = np.asarray(fvc_ratios, dtype=np.float64)
        ratio_median = float(np.median(ratio_arr))
        ratio_p25 = float(np.percentile(ratio_arr, 25))
        ratio_p75 = float(np.percentile(ratio_arr, 75))
        logger.info(
            "Unit sanity (integral(flow_true)/FVC_label): "
            f"median={ratio_median:.4f}, p25={ratio_p25:.4f}, p75={ratio_p75:.4f}, n={len(ratio_arr)}"
        )
        if 500.0 <= ratio_median <= 1500.0:
            logger.warning(
                "Possible unit mismatch: ratio is around 1000. "
                "This often means flow is mL/s while FVC label is L."
            )
        elif 0.0005 <= ratio_median <= 0.0015:
            logger.warning(
                "Possible unit mismatch: ratio is around 0.001. "
                "This often means flow is L/s while FVC label is mL."
            )

    def _load_sample_meta(sample_id):
        """Load raw time/volume metadata produced by preprocess_data.py."""
        meta_dir = getattr(config, "RAW_META_DIR", None)
        if meta_dir is None:
            return None
        meta_path = os.path.join(meta_dir, f"{sample_id}.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta if isinstance(meta, dict) else None
        except Exception:
            return None

    def _load_sample_time_and_volume(sample_id):
        """
        Priority:
        1) time/volume columns from data_pre/csv/<sample_id>.csv (time, flow, volume),
        2) raw_total_time from data_pre/meta/<sample_id>.json (equivalent-duration axis),
        3) uniform 0~3s grid fallback.
        Returns:
            (time_axis, volume_curve_from_csv_or_None)
        """
        volume_curve = None
        csv_dir = getattr(config, "CSV_DIR", None)
        if csv_dir is not None:
            sample_csv = os.path.join(csv_dir, f"{sample_id}.csv")
            if os.path.exists(sample_csv):
                try:
                    arr = np.loadtxt(sample_csv, delimiter=",", dtype=np.float64)
                    if arr.ndim == 1:
                        if arr.shape[0] >= 2:
                            arr = arr.reshape(1, -1)
                        else:
                            return uniform_time, volume_curve
                    if arr.shape[0] != seq_len or arr.shape[1] < 2:
                        return uniform_time, volume_curve
                    t = arr[:, 0].astype(np.float64)
                    if not np.all(np.isfinite(t)):
                        return uniform_time, volume_curve
                    # Ensure non-decreasing time for stable integration
                    t = np.maximum.accumulate(t)
                    if np.ptp(t) <= 1e-10:
                        return uniform_time, volume_curve

                    # If 3rd column exists, directly use it as true volume curve from data_pre
                    if arr.shape[1] >= 3:
                        v = arr[:, 2].astype(np.float64)
                        if np.all(np.isfinite(v)):
                            volume_curve = v
                    return t, volume_curve
                except Exception:
                    return uniform_time, volume_curve

        meta = _load_sample_meta(sample_id)
        if meta is not None:
            raw_total_time = float(meta.get("raw_total_time", 0.0) or 0.0)
            if raw_total_time > 1e-10:
                return np.linspace(0.0, raw_total_time, seq_len, dtype=np.float64), volume_curve
        return uniform_time, volume_curve

    rows = []
    for sid in sorted(sample_to_gt.keys()):
        gt = sample_to_gt[sid]
        pred = sample_to_pred[sid]
        fvc_label = sample_to_fvc_label.get(sid, None)
        sample_meta = _load_sample_meta(sid)

        # Priority 1: sample-specific real time axis from csv/meta
        time_axis, volume_true_csv = _load_sample_time_and_volume(sid)

        # True volume: prefer data_pre csv third column (ground truth volume)
        # Fallback to integration only when third column is unavailable.
        if volume_true_csv is not None and len(volume_true_csv) == seq_len:
            vol_true = np.asarray(volume_true_csv, dtype=np.float64)
        else:
            vol_true = np.zeros_like(gt, dtype=np.float64)
            for j in range(1, seq_len):
                dt_j = float(max(time_axis[j] - time_axis[j - 1], 0.0))
                vol_true[j] = vol_true[j - 1] + 0.5 * (gt[j] + gt[j - 1]) * dt_j

        # Pred volume: must come from model prediction (integrate predicted flow)
        vol_pred = np.zeros_like(pred, dtype=np.float64)
        if len(vol_true) > 0:
            vol_pred[0] = float(vol_true[0])
        for j in range(1, seq_len):
            dt_j = float(max(time_axis[j] - time_axis[j - 1], 0.0))
            vol_pred[j] = vol_pred[j - 1] + 0.5 * (pred[j] + pred[j - 1]) * dt_j

        # Optional fallback calibration for true volume only (never calibrate pred volume by true target)
        if volume_true_csv is None:
            calib_target = None
            if sample_meta is not None:
                raw_terminal_volume = float(sample_meta.get("raw_terminal_volume", 0.0) or 0.0)
                if raw_terminal_volume > 1e-8:
                    calib_target = raw_terminal_volume
            if calib_target is None and fvc_label is not None and fvc_label > 1e-8:
                calib_target = float(fvc_label)
            if calib_target is not None:
                end_true = float(vol_true[-1])
                if end_true > 1e-8:
                    scale = float(np.clip(calib_target / end_true, 0.1, 10.0))
                    vol_true *= scale

        rows.append({"row_id": f"{sid}_flow_true", **{point_cols[k]: float(gt[k]) for k in range(seq_len)}})
        rows.append({"row_id": f"{sid}_flow_pred", **{point_cols[k]: float(pred[k]) for k in range(seq_len)}})
        rows.append({"row_id": f"{sid}_volume_true", **{point_cols[k]: float(vol_true[k]) for k in range(seq_len)}})
        rows.append({"row_id": f"{sid}_volume_pred", **{point_cols[k]: float(vol_pred[k]) for k in range(seq_len)}})

    df_matrix = pd.DataFrame(rows, columns=["row_id"] + point_cols)
    matrix_path = os.path.join(output_dir, "results_points.csv")
    df_matrix.to_csv(matrix_path, index=False, encoding='utf-8-sig')
    logger.info(f"Point-matrix results saved to {matrix_path} (rows={len(df_matrix)})")


def main():
    """Main cross-validation function."""
    import argparse
    
    # Parse command line arguments (keep minimal; most hyperparams come from config.py)
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation Training')
    parser.add_argument('--config-file', type=str, help='Path to JSON configuration file')
    parser.add_argument('--exp', type=str, default='now', help='Experiment name (results saved to cv_results/{exp})')

    _to_bool = lambda x: str(x).lower() in ('true', '1', 'yes', 'on')

    # Data augmentation
    parser.add_argument('--spec-augment', type=_to_bool, default=True, nargs='?', const=True,
                        help='Apply SpecAugment to Mel spectrograms during training (default: true).')

    # Innovation 1: Basis-Mixture Curve Decoder
    parser.add_argument('--use-basis-mixture-curve-decoder', type=_to_bool, default=True, nargs='?', const=True,
                        help='Enable the basis-mixture curve decoder instead of the MLP regression head (default: true).')
    parser.add_argument('--basis-mixture-num-bases', type=int, default=8,
                        help='Number of learnable curve bases K (default: 8).')

    # Innovation 2: demographic encoder
    parser.add_argument('--use-enhanced-demographic', type=_to_bool, default=True, nargs='?', const=True,
                        help='Enable demographic features and EnhancedDemographicEncoder (default: true).')
    parser.add_argument('--demographic-features', type=str, default=None,
                        help='Comma-separated demographic feature list (default: gender,height,weight). '
                             'Example: "gender,height,weight"')

    # Innovation 3: physics + smoothness loss with uncertainty-weight warmup
    parser.add_argument('--use-phys-smooth-warm', type=_to_bool, default=True, nargs='?', const=True,
                        help='Enable physics constraint, smoothness loss, uncertainty weighting, and warmup (default: true). '
                             'When enabled, the main regression loss defaults to sobolev_huber.')
    parser.add_argument('--uw-warmup-epochs', type=int, default=50,
                        help='Number of uncertainty-weight warmup epochs (default: 50). '
                             'Fixed LAMBDA_* weights are used during warmup.')
    # Sobolev-Huber options, used when Innovation 3 is enabled
    parser.add_argument('--loss-type', type=str, default='sobolev_huber',
                        choices=['mse', 'l1', 'huber', 'sobolev_l1', 'sobolev_huber'],
                        help='Regression loss type for Innovation 3 (default: sobolev_huber). '
                             'If Innovation 3 is disabled, config.py LOSS_TYPE is used.')
    parser.add_argument('--sob-alpha', type=float, default=0.2,
                        help='First-derivative weight for the Sobolev loss (default: 0.2).')
    parser.add_argument('--sob-beta', type=float, default=0.05,
                        help='Second-derivative weight for the Sobolev loss (default: 0.05).')
    parser.add_argument('--sob-delta', type=float, default=1.0,
                        help='Huber threshold for the Sobolev-Huber loss (default: 1.0).')
    
    args = parser.parse_args()
    
    # Load from JSON file if provided (updates Config class)
    if args.config_file and os.path.exists(args.config_file):
        config_dict = Config.load_json(args.config_file)
        Config.update_from_dict(config_dict)
        logger.info(f"Loaded configuration from {args.config_file}")
    
    # Update Config class from command line arguments
    # This modifies the Config class itself, so all subsequent uses will use updated values
    Config.update_from_args(args)

    # Apply data augmentation switch
    Config.SPEC_AUGMENT = bool(getattr(args, 'spec_augment', True))

    # Apply innovation switches
    Config.USE_ENHANCED_DEMOGRAPHIC_ENCODER = bool(getattr(args, 'use_enhanced_demographic', False))
    Config.USE_SPECTRO_TEMPORAL_CONDITION_ENCODER = bool(getattr(args, 'use_basis_mixture_curve_decoder', False))
    Config.BASIS_MIXTURE_NUM_BASES = int(getattr(args, 'basis_mixture_num_bases', 8) or 8)
    use_phys_smooth_warm = bool(getattr(args, 'use_phys_smooth_warm', False))
    if use_phys_smooth_warm:
        Config.USE_PHYS_SMOOTH_LOSS = True
        # allow 0 to disable warmup explicitly
        _uw = getattr(args, 'uw_warmup_epochs', 50)
        Config.UW_WARMUP_EPOCHS = 50 if _uw is None else int(_uw)
    else:
        Config.USE_PHYS_SMOOTH_LOSS = False
        _uw = getattr(args, 'uw_warmup_epochs', 50)
        Config.UW_WARMUP_EPOCHS = 50 if _uw is None else int(_uw)

    # Apply regression loss switches (Innovation 3 only)
    if use_phys_smooth_warm:
        # default under innovation-3: sobolev_huber
        _loss_type = getattr(args, 'loss_type', None)
        Config.LOSS_TYPE = 'sobolev_huber' if _loss_type is None else str(_loss_type)

        _sob_a = getattr(args, 'sob_alpha', None)
        if _sob_a is not None:
            Config.SOB_ALPHA = float(_sob_a)
        _sob_b = getattr(args, 'sob_beta', None)
        if _sob_b is not None:
            Config.SOB_BETA = float(_sob_b)
        _sob_d = getattr(args, 'sob_delta', None)
        if _sob_d is not None:
            Config.SOB_DELTA = float(_sob_d)

    if Config.USE_ENHANCED_DEMOGRAPHIC_ENCODER:
        Config.USE_DEMOGRAPHIC = True
        if args.demographic_features is not None:
            features = [x.strip() for x in str(args.demographic_features).split(',') if x.strip()]
            Config.DEMOGRAPHIC_FEATURES = features if features else ['gender', 'height', 'weight']
        else:
            Config.DEMOGRAPHIC_FEATURES = ['gender', 'height', 'weight']
    else:
        Config.USE_DEMOGRAPHIC = False
        Config.DEMOGRAPHIC_FEATURES = []

    # Always prefer CUDA if available, otherwise CPU
    # (Override any misconfigured Config.DEVICE / CLI --device)
    Config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    Config.PIN_MEMORY = True if torch.cuda.is_available() else False

    # Ensure output directory exists and attach file logging.
    # This makes warmup/uncertainty-weighting behavior auditable after training.
    Config.EXP = str(getattr(args, 'exp', getattr(Config, 'EXP', 'now')) or 'now')
    output_dir = os.path.join('cv_results', Config.EXP)
    os.makedirs(output_dir, exist_ok=True)
    try:
        log_path = os.path.join(output_dir, 'train.log')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == fh.baseFilename for h in root.handlers):
            root.addHandler(fh)
        logger.info(f"Logging to file: {log_path}")
    except Exception as e:
        logger.warning(f"Failed to attach FileHandler for logging: {e}")

    # Persist a config snapshot for reproducibility/debugging.
    try:
        # Config.to_dict() may include classmethods; filter them out for JSON snapshot.
        cfg_path = os.path.join(output_dir, 'config_snapshot.json')
        cfg = {
            k: v
            for k, v in Config.to_dict().items()
            if (not callable(v)) and (not isinstance(v, classmethod))
        }
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info(f"Config snapshot saved: {cfg_path}")
    except Exception as e:
        logger.warning(f"Failed to save config snapshot: {e}")
    
    # Use the global Config class (now updated with command line args)
    # No need to create a new instance
    
    # Record start time
    import time
    start_time = time.time()
    
    # Set random seed at the very beginning
    set_seed(Config.RANDOM_SEED)
    
    logger.info("="*70)
    logger.info("5-Fold Cross-Validation Training")
    logger.info("="*70)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Model: {getattr(Config, 'MODEL', 'direct')}")
    logger.info(f"Random seed: {Config.RANDOM_SEED}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"Deterministic algorithms: Enabled")
    logger.info(f"Epochs per fold: {Config.NUM_EPOCHS}")
    logger.info(f"Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")
    
    # Get all sample IDs
    # IMPORTANT: Sort the list to ensure deterministic order!
    mel_files = sorted([f.replace('.npy', '') for f in os.listdir(Config.MEL_DIR) if f.endswith('.npy')])
    
    labels_df = pd.read_csv(Config.LABEL_FILE)
    label_dict = {f"{int(row['id']):04d}": True for _, row in labels_df.iterrows()}
    
    valid_samples = []
    for sid in mel_files:
        subject_id = sid.split('_')[0]
        if subject_id in label_dict:
            valid_samples.append(sid)
    
    logger.info(f"Total valid samples: {len(valid_samples)}")
    
    # Log subject count, sample count, sex distribution, and model parameters before training
    log_training_summary(valid_samples, labels_df, config=Config)
    
    # Create K-fold splits
    logger.info("\nCreating 5-fold splits...")
    folds = create_kfold_splits(valid_samples, n_folds=5, random_seed=Config.RANDOM_SEED)
    
    # Output directory (use EXP parameter)
    exp_name = getattr(Config, 'EXP', 'now')
    output_dir = os.path.join('cv_results', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Store all fold metrics
    all_fold_metrics = []
    
    # Store all sample-level data for detailed analysis
    all_sample_data = []
    
    # Train and test each fold
    # Use the global Config class (updated from command line args if any)
    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        # Train (returns best model path for this fold)
        checkpoint_path = train_fold(fold_idx, train_ids, test_ids, output_dir, config=Config)
        
        # Test
        fold_metrics, fold_sample_data = test_fold(fold_idx, test_ids, checkpoint_path, output_dir, config=Config)
        
        all_fold_metrics.append(fold_metrics)
        all_sample_data.append(fold_sample_data)
    
    # Compute mean and std across folds
    logger.info("\n" + "="*70)
    logger.info("Cross-Validation Results (Mean ± Std)")
    logger.info("="*70)
    
    metric_names = list(all_fold_metrics[0].keys())
    final_results = {}
    
    for metric_name in metric_names:
        values = [fold_metrics[metric_name] for fold_metrics in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_results[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'values': [float(v) for v in values]
        }
        logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    # Repeat the training summary after completion and add single-sample inference time
    logger.info("\n" + "="*70)
    logger.info("Post-Training Summary")
    logger.info("="*70)
    log_training_summary(valid_samples, labels_df, config=Config)
    if 'inference_ms_per_sample' in final_results:
        inf = final_results['inference_ms_per_sample']
        logger.info(f"Single-sample inference time (ms/sample): {inf['mean']:.4f} ± {inf['std']:.4f}")
    logger.info("="*70)
    
    # Calculate total time
    end_time = time.time()
    total_seconds = end_time - start_time
    total_minutes = int(total_seconds // 60)
    remaining_seconds = int(total_seconds % 60)
    
    # Add metadata to results
    final_results['_metadata'] = {
        'total_time_seconds': float(total_seconds),
        'total_time_formatted': f"{total_minutes} min {remaining_seconds} s",
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': Config.DEVICE,
        'epochs_per_fold': Config.NUM_EPOCHS,
        'random_seed': Config.RANDOM_SEED,
        'inference_ms_per_sample_unit': 'ms/sample'
    }
    
    # Save results to JSON
    json_path = os.path.join(output_dir, 'cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")
    
    # Save results to TXT
    txt_path = os.path.join(output_dir, 'cv_results.txt')
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("5-Fold Cross-Validation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {Config.DEVICE}\n")
        f.write(f"Epochs per fold: {Config.NUM_EPOCHS}\n")
        f.write(f"Total samples: {len(valid_samples)}\n")
        f.write(f"Total time: {total_minutes} min {remaining_seconds} s\n\n")
        
        f.write("Results (Mean ± Std):\n")
        f.write("-"*70 + "\n")
        for metric_name, result in final_results.items():
            if metric_name != '_metadata':  # Skip metadata
                if metric_name == 'inference_ms_per_sample':
                    f.write(f"{metric_name} (ms/sample): {result['mean']:.4f} ± {result['std']:.4f}\n")
                else:
                    f.write(f"{metric_name}: {result['mean']:.4f} ± {result['std']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Individual Fold Results:\n")
        f.write("="*70 + "\n")
        for fold_idx, fold_metrics in enumerate(all_fold_metrics):
            f.write(f"\nFold {fold_idx + 1}:\n")
            for metric_name, value in fold_metrics.items():
                if metric_name == 'inference_ms_per_sample':
                    f.write(f"  {metric_name} (ms/sample): {value:.4f}\n")
                else:
                    f.write(f"  {metric_name}: {value:.4f}\n")
    
    logger.info(f"Results saved to {txt_path}")
    
    # Generate sample-level results Excel file
    logger.info("\nGenerating sample-level results Excel file...")
    generate_sample_results(all_sample_data, output_dir, config=Config)
    
    # Generate sample point matrix CSV (4 rows per sample, 60 columns for points)
    logger.info("\nGenerating point-matrix results CSV...")
    generate_sample_point_matrix_csv(all_sample_data, output_dir, config=Config)
    
    logger.info("\n" + "="*70)
    logger.info("Cross-Validation Completed!")
    logger.info(f"Total time: {total_minutes} min {remaining_seconds} s")
    logger.info("="*70)


if __name__ == '__main__':
    main()

