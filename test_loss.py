import torch
from informer.loss_functions import CombinedLoss, DirectionalLoss, AsymmetricLoss

print("Testing loss functions on MPS device...")

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Create mock data
batch_size = 8
sequence_length = 24
features = 5
pred = torch.randn(batch_size, sequence_length, features, device=device)
target = torch.randn(batch_size, sequence_length, features, device=device)

print(f"Prediction tensor shape: {pred.shape}")
print(f"Target tensor shape: {target.shape}")

# Test MSE + MAE combo
print("\nTesting CombinedLoss with MSE and MAE...")
criterion1 = CombinedLoss(mse_weight=0.7, mae_weight=0.3, directional_weight=0.0, asymmetric_weight=0.0)
loss1 = criterion1(pred, target)
print(f"Combined MSE+MAE Loss: {loss1.item():.6f}")

# Test with directional component
print("\nTesting CombinedLoss with directional component...")
criterion2 = CombinedLoss(mse_weight=0.5, mae_weight=0.3, directional_weight=0.2, asymmetric_weight=0.0)
loss2 = criterion2(pred, target)
print(f"Combined Loss with directional component: {loss2.item():.6f}")

# Test with asymmetric component
print("\nTesting CombinedLoss with asymmetric component...")
criterion3 = CombinedLoss(mse_weight=0.5, mae_weight=0.3, directional_weight=0.0, asymmetric_weight=0.2,
                         asymmetric_alpha=1.5, asymmetric_beta=1.0)
loss3 = criterion3(pred, target)
print(f"Combined Loss with asymmetric component: {loss3.item():.6f}")

# Test with all components
print("\nTesting CombinedLoss with all components...")
criterion4 = CombinedLoss(mse_weight=0.4, mae_weight=0.3, directional_weight=0.15, asymmetric_weight=0.15,
                         asymmetric_alpha=1.5, asymmetric_beta=1.0)
loss4 = criterion4(pred, target)
print(f"Combined Loss with all components: {loss4.item():.6f}")

print("\nAll tests completed successfully!") 