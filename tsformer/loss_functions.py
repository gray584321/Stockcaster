import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Directional Loss penalizes the model more heavily when it predicts the wrong direction
    of stock price movement. Especially important for financial time series forecasting.
    """
    def __init__(self, mse_weight=0.7, directional_weight=0.3, close_price_index=0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        self.close_price_index = close_price_index  # Index of close price in feature vector
        
    def forward(self, pred, target):
        # Calculate standard MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate directional loss - penalize wrong directions
        # We care most about the close price (assumed to be at close_price_index)
        pred_diff = pred[:, 1:, self.close_price_index] - pred[:, :-1, self.close_price_index]
        target_diff = target[:, 1:, self.close_price_index] - target[:, :-1, self.close_price_index]
        
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
    
    In financial time series, this is useful when the cost of underestimating a price
    is different from the cost of overestimating it.
    """
    def __init__(self, alpha=1.5, beta=1.0, feature_weights=None):
        super().__init__()
        # alpha > beta: under-predictions are penalized more
        # alpha < beta: over-predictions are penalized more
        self.alpha = alpha
        self.beta = beta
        # Optional weights for different features
        self.feature_weights = feature_weights
        
    def forward(self, pred, target):
        # Calculate errors
        errors = pred - target
        
        # Apply asymmetric weighting
        weighted_errors = torch.where(
            errors < 0,
            self.alpha * torch.pow(errors, 2),  # Under-predictions
            self.beta * torch.pow(errors, 2)    # Over-predictions
        )
        
        # Apply feature weights if provided
        if self.feature_weights is not None:
            feature_weights = self.feature_weights.to(weighted_errors.device)
            weighted_errors = weighted_errors * feature_weights.view(1, 1, -1)
        
        return weighted_errors.mean()


class CombinedLoss(nn.Module):
    """
    A combined loss function that mixes MSE with MAE and potentially adds
    directional or asymmetric components for stock price prediction.
    
    This is particularly useful for financial time series forecasting where
    different aspects of the prediction may have different importance.
    """
    def __init__(self, mse_weight=0.7, mae_weight=0.3, directional_weight=0.0, asymmetric_weight=0.0, 
                 asymmetric_alpha=1.5, asymmetric_beta=1.0, feature_weights=None, close_price_index=0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.directional_loss = DirectionalLoss(close_price_index=close_price_index) if directional_weight > 0 else None
        self.asymmetric_loss = AsymmetricLoss(alpha=asymmetric_alpha, beta=asymmetric_beta, 
                                             feature_weights=feature_weights) if asymmetric_weight > 0 else None
        
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


class TemporalWeightedLoss(nn.Module):
    """
    Loss function that applies higher weights to later time steps in the prediction,
    making the model focus more on near-future predictions.
    
    This is useful for financial time series where accuracy of immediate future
    predictions may be more important than longer-term predictions.
    """
    def __init__(self, base_criterion=None, gamma=1.10):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss()
        self.gamma = gamma  # Controls how quickly weights increase with time
        
    def forward(self, pred, target):
        # Create temporal weights that increase with time
        seq_len = pred.size(1)
        time_indices = torch.arange(seq_len, device=pred.device, dtype=torch.float32)
        weights = torch.pow(self.gamma, time_indices)
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted loss across time dimension
        loss = 0.0
        for t in range(seq_len):
            step_loss = self.base_criterion(pred[:, t], target[:, t])
            loss += weights[t] * step_loss
            
        return loss 