"""Loss functions for curve regression, physical constraints, and smoothness."""
import torch
import torch.nn.functional as F


def regression_curve_loss(flow_pred, flow_gt, loss_type='l1', sob_alpha=0.2, sob_beta=0.05, sob_delta=1.0):
    """
    Supervised regression loss for direct curve prediction.
    
    Args:
        flow_pred: Predicted flow curve (B, 60) or (B, 1, 60)
        flow_gt: Ground truth flow curve (B, 60)
        loss_type: 'mse', 'l1' (MAE), 'huber', 'sobolev_l1', or 'sobolev_huber'
    
    Returns:
        Scalar loss value
    """
    if flow_pred.dim() == 3:
        flow_pred = flow_pred.squeeze(1)
    if loss_type == 'mse':
        loss = (flow_pred - flow_gt) ** 2
        return loss.mean()
    if loss_type == 'l1':
        return torch.abs(flow_pred - flow_gt).mean()
    if loss_type == 'huber':
        return F.smooth_l1_loss(flow_pred, flow_gt)
    if loss_type == 'sobolev_l1':
        # Value term
        l_val = torch.mean(torch.abs(flow_pred - flow_gt))
        # First derivative term
        d1_pred = flow_pred[:, 1:] - flow_pred[:, :-1]
        d1_gt = flow_gt[:, 1:] - flow_gt[:, :-1]
        l_d1 = torch.mean(torch.abs(d1_pred - d1_gt))
        # Second derivative term
        d2_pred = d1_pred[:, 1:] - d1_pred[:, :-1]
        d2_gt = d1_gt[:, 1:] - d1_gt[:, :-1]
        l_d2 = torch.mean(torch.abs(d2_pred - d2_gt))
        return l_val + float(sob_alpha) * l_d1 + float(sob_beta) * l_d2
    if loss_type == 'sobolev_huber':
        # Value term uses Huber for better robustness/stability
        l_val = F.smooth_l1_loss(flow_pred, flow_gt, beta=float(sob_delta))
        # Derivative terms keep L1 to preserve trend/shape constraints
        d1_pred = flow_pred[:, 1:] - flow_pred[:, :-1]
        d1_gt = flow_gt[:, 1:] - flow_gt[:, :-1]
        l_d1 = torch.mean(torch.abs(d1_pred - d1_gt))
        d2_pred = d1_pred[:, 1:] - d1_pred[:, :-1]
        d2_gt = d1_gt[:, 1:] - d1_gt[:, :-1]
        l_d2 = torch.mean(torch.abs(d2_pred - d2_gt))
        return l_val + float(sob_alpha) * l_d1 + float(sob_beta) * l_d2
    raise ValueError(f"Unknown loss type: {loss_type}")


def physical_constraint_loss(flow_pred, fev1_gt, fvc_gt, pef_gt, dt=None,
                             fev1_weight=1.0, fvc_weight=1.0, pef_weight=1.0):
    """
    Compute physical constraint loss based on pulmonary function metrics.
    
    Note: Time points are uniformly distributed from 0 to 3s with inclusive endpoints.
    If there are N points, the time step between adjacent points is dt = 3 / (N-1).
    - FEV1: integral from 0 to exactly 1s (with linear interpolation at boundary)
    - FVC: integral from 0 to 3s (all points)
    - PEF: maximum flow value
    
    Args:
        flow_pred: Predicted flow curve (B, 60)
        fev1_gt: Ground truth FEV1 (B,)
        fvc_gt: Ground truth FVC (B,)
        pef_gt: Ground truth PEF (B,)
        dt: Time step between adjacent points. If None, inferred as 3/(N-1).
        fev1_weight: Weight for FEV1 loss (default 1.0)
        fvc_weight: Weight for FVC loss (default 1.0)
        pef_weight: Weight for PEF loss (default 1.0)
    
    Returns:
        Scalar loss value
    """
    # Ensure flow_pred is 2D
    if flow_pred.dim() == 3:
        flow_pred = flow_pred.squeeze(1)  # (B, 60)
    
    # Infer dt from the number of points (N includes both endpoints).
    if dt is None:
        seq_len = int(flow_pred.size(1))
        dt = 3.0 / max(seq_len - 1, 1)

    # FEV1: integral from 0 to exactly 1s (with linear interpolation at boundary).
    # 1s falls between grid points, so interpolate and add the partial interval.
    idx_1s = int(1.0 / dt)
    t_left = idx_1s * dt
    alpha_1s = (1.0 - t_left) / dt
    f_at_1s = flow_pred[:, idx_1s] * (1.0 - alpha_1s) + flow_pred[:, idx_1s + 1] * alpha_1s
    fev1_pred = (torch.trapezoid(flow_pred[:, :idx_1s + 1], dx=dt, dim=1)
                 + 0.5 * (flow_pred[:, idx_1s] + f_at_1s) * (1.0 - t_left))
    
    # FVC: integral from 0 to 3s (all 60 points)
    # Time points are uniformly distributed, so all 60 points = 0-3s
    fvc_pred = torch.trapezoid(flow_pred, dx=dt, dim=1)
    
    # PEF: maximum flow value
    pef_pred = torch.max(flow_pred, dim=1)[0]  # (B,)
    
    # Compute L1 loss (MAE) for each metric
    loss_fev1 = F.l1_loss(fev1_pred, fev1_gt)
    loss_fvc = F.l1_loss(fvc_pred, fvc_gt)
    loss_pef = F.l1_loss(pef_pred, pef_gt)
    
    # Apply weights to each component
    return fev1_weight * loss_fev1 + fvc_weight * loss_fvc + pef_weight * loss_pef


def smoothness_loss(flow_pred, order=2):
    """
    Compute smoothness loss based on higher-order derivatives.
    Encourages smooth predictions by penalizing large second derivatives.
    
    Args:
        flow_pred: Predicted flow curve (B, 60)
        order: Order of derivative to penalize (1 for first derivative, 2 for second derivative)
    
    Returns:
        Scalar smoothness loss value
    """
    # Ensure flow_pred is 2D
    if flow_pred.dim() == 3:
        flow_pred = flow_pred.squeeze(1)  # (B, 60)
    
    if order == 1:
        # Penalize first derivative (gradient)
        # Compute differences between consecutive points
        diff = flow_pred[:, 1:] - flow_pred[:, :-1]  # (B, 59)
        # L2 norm of differences
        loss = torch.mean(diff ** 2)
    elif order == 2:
        # Penalize second derivative (curvature)
        # Compute second differences
        first_diff = flow_pred[:, 1:] - flow_pred[:, :-1]  # (B, 59)
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]  # (B, 58)
        # L2 norm of second differences
        loss = torch.mean(second_diff ** 2)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 1 or 2.")
    
    return loss



