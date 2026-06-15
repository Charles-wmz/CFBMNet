"""
Evaluation metrics for flow prediction.
"""
import torch
import numpy as np


def compute_icc(values1, values2, icc_type='ICC(3,1)'):
    """
    Compute Intraclass Correlation Coefficient (ICC) between two sets of measurements.

    ICC(3,1) is a two-way mixed effects model, consistency case, single raters.
    This is appropriate when comparing predictions to ground truth.

    Args:
        values1: First set of measurements (e.g., predictions)
        values2: Second set of measurements (e.g., ground truth)
        icc_type: Type of ICC ('ICC(3,1)' or 'ICC(2,1)')

    Returns:
        ICC value
    """
    values1 = np.asarray(values1).flatten()
    values2 = np.asarray(values2).flatten()

    n = len(values1)
    if n == 0:
        return float('nan')

    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    grand_mean = (mean1 + mean2) / 2.0

    ss_between = n * ((mean1 - grand_mean) ** 2 + (mean2 - grand_mean) ** 2)
    ss_total = np.sum((values1 - grand_mean) ** 2) + np.sum((values2 - grand_mean) ** 2)

    ms_between = ss_between
    ms_error = (np.sum((values1 - mean1) ** 2) + np.sum((values2 - mean2) ** 2)) / (2 * n)

    if ms_error == 0:
        return float('nan')

    icc = (ms_between - ms_error) / (ms_between + ms_error)

    if icc_type == 'ICC(2,1)':
        ms_between = n * np.var([mean1, mean2])
        ms_within = ms_error
        icc = (ms_between - ms_within) / (ms_between + (n - 1) * ms_within)

    return float(np.clip(icc, -1.0, 1.0))


class MetricsCalculator:
    """
    Calculate evaluation metrics for flow prediction.
    """
    
    @staticmethod
    def compute_flow_metrics(pred, gt):
        """
        Compute metrics for flow curve prediction.
        
        Args:
            pred: Predicted flow curves (N, 60) or (N,) tensor/array
            gt: Ground truth flow curves (N, 60) or (N,) tensor/array
        
        Returns:
            Dictionary with MAE, RMSE, MAPE
        """
        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        
        # Ensure 2D
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if gt.ndim == 1:
            gt = gt.reshape(-1, 1)
        
        # MAE
        mae = np.mean(np.abs(pred - gt))
        
        # RMSE
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        
        # MAPE (avoid division by zero and filter out near-zero values)
        # Only calculate MAPE for points where |gt| > threshold
        threshold = 0.1  # Ignore points with flow < 0.1 L/s
        mask = np.abs(gt) > threshold
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((pred[mask] - gt[mask]) / gt[mask])) * 100
        else:
            mape = 0.0  # All values too small, MAPE not meaningful
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    @staticmethod
    def compute_pulmonary_metrics(flow_pred, labels_gt, dt=None):
        """
        Compute pulmonary function metrics (FEV1, FVC, PEF).
        
        Args:
            flow_pred: Predicted flow curves (N, 60)
            labels_gt: Ground truth labels (N, 3) [FEV1, FVC, PEF]
            dt: Time step between adjacent points. If None, inferred as 3/(N-1).
        
        Returns:
            Dictionary with metrics for FEV1, FVC, PEF (each with MAE, RMSE, MAPE)
        """
        # Convert to numpy if needed
        if isinstance(flow_pred, torch.Tensor):
            flow_pred = flow_pred.detach().cpu().numpy()
        if isinstance(labels_gt, torch.Tensor):
            labels_gt = labels_gt.detach().cpu().numpy()
        
        # Infer dt from the number of points (N includes both endpoints).
        if dt is None:
            seq_len = int(flow_pred.shape[1])
            dt = 3.0 / max(seq_len - 1, 1)

        # FEV1: integral from 0 to exactly 1s (with linear interpolation at boundary)
        idx_1s = int(1.0 / dt)
        t_left = idx_1s * dt
        alpha_1s = (1.0 - t_left) / dt
        f_at_1s = flow_pred[:, idx_1s] * (1.0 - alpha_1s) + flow_pred[:, idx_1s + 1] * alpha_1s
        fev1_pred = (np.trapz(flow_pred[:, :idx_1s + 1], dx=dt, axis=1)
                     + 0.5 * (flow_pred[:, idx_1s] + f_at_1s) * (1.0 - t_left))
        fev1_gt = labels_gt[:, 0]
        
        fev1_mae = np.mean(np.abs(fev1_pred - fev1_gt))
        fev1_rmse = np.sqrt(np.mean((fev1_pred - fev1_gt) ** 2))
        fev1_mape = np.mean(np.abs((fev1_pred - fev1_gt) / (fev1_gt + 1e-8))) * 100
        
        # FVC: integral from 0 to 3s (all 60 points)
        fvc_pred = np.trapz(flow_pred, dx=dt, axis=1)
        fvc_gt = labels_gt[:, 1]
        
        fvc_mae = np.mean(np.abs(fvc_pred - fvc_gt))
        fvc_rmse = np.sqrt(np.mean((fvc_pred - fvc_gt) ** 2))
        fvc_mape = np.mean(np.abs((fvc_pred - fvc_gt) / (fvc_gt + 1e-8))) * 100
        
        # PEF: maximum flow
        pef_pred = np.max(flow_pred, axis=1)
        pef_gt = labels_gt[:, 2]

        pef_mae = np.mean(np.abs(pef_pred - pef_gt))
        pef_rmse = np.sqrt(np.mean((pef_pred - pef_gt) ** 2))
        pef_mape = np.mean(np.abs((pef_pred - pef_gt) / (pef_gt + 1e-8))) * 100

        # FEV1/FVC ratio: computed from predicted FEV1 and FVC
        fev1_fvc_pred = fev1_pred / (fvc_pred + 1e-8)
        fev1_fvc_gt = fev1_gt / (fvc_gt + 1e-8)

        fev1_fvc_mae = np.mean(np.abs(fev1_fvc_pred - fev1_fvc_gt))
        fev1_fvc_rmse = np.sqrt(np.mean((fev1_fvc_pred - fev1_fvc_gt) ** 2))
        fev1_fvc_mape = np.mean(np.abs((fev1_fvc_pred - fev1_fvc_gt) / (fev1_fvc_gt + 1e-8))) * 100

        return {
            'fev1_mae': float(fev1_mae),
            'fev1_rmse': float(fev1_rmse),
            'fev1_mape': float(fev1_mape),
            'fvc_mae': float(fvc_mae),
            'fvc_rmse': float(fvc_rmse),
            'fvc_mape': float(fvc_mape),
            'pef_mae': float(pef_mae),
            'pef_rmse': float(pef_rmse),
            'pef_mape': float(pef_mape),
            'fev1_fvc_mae': float(fev1_fvc_mae),
            'fev1_fvc_rmse': float(fev1_fvc_rmse),
            'fev1_fvc_mape': float(fev1_fvc_mape),
            'fev1_pred': fev1_pred,
            'fvc_pred': fvc_pred,
            'pef_pred': pef_pred,
            'fev1_gt': fev1_gt,
            'fvc_gt': fvc_gt,
            'pef_gt': pef_gt,
        }
    
    @staticmethod
    def compute_all_metrics(flow_pred, flow_gt, labels_gt):
        """
        Compute all metrics.
        
        Args:
            flow_pred: Predicted flow curves (N, 60)
            flow_gt: Ground truth flow curves (N, 60)
            labels_gt: Ground truth labels (N, 3) [FEV1, FVC, PEF]
        
        Returns:
            Dictionary with all metrics in order: Flow -> FVC -> FEV1 -> PEF
        """
        flow_metrics = MetricsCalculator.compute_flow_metrics(flow_pred, flow_gt)
        pulmonary_metrics = MetricsCalculator.compute_pulmonary_metrics(flow_pred, labels_gt)
        
        # Return in specific order: Flow -> FVC -> FEV1 -> PEF
        from collections import OrderedDict
        ordered_metrics = OrderedDict()
        
        # Flow metrics first
        ordered_metrics['mae'] = flow_metrics['mae']
        ordered_metrics['rmse'] = flow_metrics['rmse']
        ordered_metrics['mape'] = flow_metrics['mape']
        
        # FVC metrics
        ordered_metrics['fvc_mae'] = pulmonary_metrics['fvc_mae']
        ordered_metrics['fvc_rmse'] = pulmonary_metrics['fvc_rmse']
        ordered_metrics['fvc_mape'] = pulmonary_metrics['fvc_mape']
        
        # FEV1 metrics
        ordered_metrics['fev1_mae'] = pulmonary_metrics['fev1_mae']
        ordered_metrics['fev1_rmse'] = pulmonary_metrics['fev1_rmse']
        ordered_metrics['fev1_mape'] = pulmonary_metrics['fev1_mape']
        
        # PEF metrics
        ordered_metrics['pef_mae'] = pulmonary_metrics['pef_mae']
        ordered_metrics['pef_rmse'] = pulmonary_metrics['pef_rmse']
        ordered_metrics['pef_mape'] = pulmonary_metrics['pef_mape']

        # FEV1/FVC ratio metrics
        ordered_metrics['fev1_fvc_mae'] = pulmonary_metrics['fev1_fvc_mae']
        ordered_metrics['fev1_fvc_rmse'] = pulmonary_metrics['fev1_fvc_rmse']
        ordered_metrics['fev1_fvc_mape'] = pulmonary_metrics['fev1_fvc_mape']

        # ICC (Intraclass Correlation Coefficient) for FEV1, FVC, PEF
        fev1_icc = compute_icc(pulmonary_metrics['fev1_pred'], pulmonary_metrics['fev1_gt'])
        fvc_icc = compute_icc(pulmonary_metrics['fvc_pred'], pulmonary_metrics['fvc_gt'])
        pef_icc = compute_icc(pulmonary_metrics['pef_pred'], pulmonary_metrics['pef_gt'])
        fev1_fvc_pred = pulmonary_metrics['fev1_pred'] / (pulmonary_metrics['fvc_pred'] + 1e-8)
        fev1_fvc_gt = pulmonary_metrics['fev1_gt'] / (pulmonary_metrics['fvc_gt'] + 1e-8)
        fev1_fvc_icc = compute_icc(fev1_fvc_pred, fev1_fvc_gt)

        ordered_metrics['fev1_icc'] = fev1_icc
        ordered_metrics['fvc_icc'] = fvc_icc
        ordered_metrics['pef_icc'] = pef_icc
        ordered_metrics['fev1_fvc_icc'] = fev1_fvc_icc

        return ordered_metrics
    
    @staticmethod
    def print_metrics(metrics, prefix=''):
        """
        Print metrics in a formatted way.
        Order: Flow -> FVC -> FEV1 -> PEF
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for printing (e.g., 'Train', 'Val', 'Test')
        """
        print(f"\n{prefix} Metrics:")
        print("=" * 70)
        
        # Flow Curve Metrics
        print(f"Flow Curve Metrics:")
        print(f"  MAE:  {metrics.get('mae', 0):.4f} L/s")
        print(f"  RMSE: {metrics.get('rmse', 0):.4f} L/s")
        print(f"  MAPE: {metrics.get('mape', 0):.2f}%")
        
        # FVC Metrics
        print(f"\nFVC (Forced Vital Capacity) Metrics:")
        print(f"  MAE:  {metrics.get('fvc_mae', 0):.4f} L")
        print(f"  RMSE: {metrics.get('fvc_rmse', 0):.4f} L")
        print(f"  MAPE: {metrics.get('fvc_mape', 0):.2f}%")
        
        # FEV1 Metrics
        print(f"\nFEV1 (Forced Expiratory Volume in 1s) Metrics:")
        print(f"  MAE:  {metrics.get('fev1_mae', 0):.4f} L")
        print(f"  RMSE: {metrics.get('fev1_rmse', 0):.4f} L")
        print(f"  MAPE: {metrics.get('fev1_mape', 0):.2f}%")
        
        # PEF Metrics
        print(f"\nPEF (Peak Expiratory Flow) Metrics:")
        print(f"  MAE:  {metrics.get('pef_mae', 0):.4f} L/s")
        print(f"  RMSE: {metrics.get('pef_rmse', 0):.4f} L/s")
        print(f"  MAPE: {metrics.get('pef_mape', 0):.2f}%")

        # FEV1/FVC Metrics
        print(f"\nFEV1/FVC Ratio Metrics:")
        print(f"  MAE:  {metrics.get('fev1_fvc_mae', 0):.4f}")
        print(f"  RMSE: {metrics.get('fev1_fvc_rmse', 0):.4f}")
        print(f"  MAPE: {metrics.get('fev1_fvc_mape', 0):.2f}%")

        # ICC Metrics
        print(f"\nIntraclass Correlation Coefficient (ICC):")
        print(f"  FEV1 ICC:       {metrics.get('fev1_icc', 0):.4f}")
        print(f"  FVC ICC:        {metrics.get('fvc_icc', 0):.4f}")
        print(f"  PEF ICC:        {metrics.get('pef_icc', 0):.4f}")
        print(f"  FEV1/FVC ICC:   {metrics.get('fev1_fvc_icc', 0):.4f}")

        print("=" * 70)
