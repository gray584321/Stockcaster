import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Directional Loss penalizes the model more heavily when it predicts the wrong direction
    of stock price movement.
    """
    def __init__(self, mse_weight=0.7, directional_weight=0.3):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        
    def forward(self, pred, target):
        # Calculate standard MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate directional loss - penalize wrong directions
        # We care most about the close price (assumed to be feature 0)
        pred_diff = pred[:, 1:, 0] - pred[:, :-1, 0]
        target_diff = target[:, 1:, 0] - target[:, :-1, 0]
        
        # Sign mismatch (wrong direction) is penalized
        direction_match = torch.sign(pred_diff) * torch.sign(target_diff)
        # Values < 0 mean wrong direction prediction
        directional_penalty = torch.clamp(-direction_match, min=0)
        # Mean over all timesteps and samples
        directional_loss = directional_penalty.mean()
        
        # Combine the losses
        combined_loss = self.mse_weight * mse + self.directional_weight * directional_loss
        return combined_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss penalizes under-predictions (or over-predictions) more heavily,
    depending on what's more important for the specific trading strategy.
    """
    def __init__(self, alpha=1.5, beta=1.0):
        super().__init__()
        # alpha > beta: under-predictions are penalized more
        # alpha < beta: over-predictions are penalized more
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        # Calculate errors
        errors = pred - target
        
        # Apply asymmetric weighting
        weighted_errors = torch.where(
            errors < 0,
            self.alpha * torch.pow(errors, 2),  # Under-predictions
            self.beta * torch.pow(errors, 2)    # Over-predictions
        )
        
        return weighted_errors.mean()


class CombinedLoss(nn.Module):
    """
    A combined loss function that mixes MSE with MAE and potentially adds
    directional or asymmetric components for stock price prediction.
    """
    def __init__(self, mse_weight=0.7, mae_weight=0.3, directional_weight=0.0, asymmetric_weight=0.0, 
                 asymmetric_alpha=1.5, asymmetric_beta=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.directional_loss = DirectionalLoss() if directional_weight > 0 else None
        self.asymmetric_loss = AsymmetricLoss(alpha=asymmetric_alpha, beta=asymmetric_beta) if asymmetric_weight > 0 else None
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.directional_weight = directional_weight
        self.asymmetric_weight = asymmetric_weight
        
        # Ensure weights sum to 1
        total_weight = mse_weight + mae_weight + directional_weight + asymmetric_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.mse_weight /= total_weight
            self.mae_weight /= total_weight
            self.directional_weight /= total_weight
            self.asymmetric_weight /= total_weight
        
    def forward(self, pred, target):
        loss = self.mse_weight * self.mse_loss(pred, target)
        
        if self.mae_weight > 0:
            loss += self.mae_weight * self.mae_loss(pred, target)
            
        if self.directional_weight > 0 and self.directional_loss is not None:
            # Only use the directional component, not the MSE part
            dir_loss = self.directional_loss(pred, target)
            dir_component = dir_loss - self.mse_loss(pred, target) * self.directional_loss.mse_weight
            loss += self.directional_weight * dir_component
            
        if self.asymmetric_weight > 0 and self.asymmetric_loss is not None:
            loss += self.asymmetric_weight * self.asymmetric_loss(pred, target)
            
        return loss 