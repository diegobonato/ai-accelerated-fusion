import torch
import torch.nn as nn
import torch.nn.functional as F


data = torch.randn(1, 3, 224, 224)
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


# Move to cuda.
data = data.to("cuda")
model = model.to("cuda")


# Forward pass
output = model(data)
print("Output shape:", output.shape)
# Loss function
criterion = nn.CrossEntropyLoss()
# Dummy target
target = torch.randint(0, 10, (1,)).to("cuda")
# Compute loss
loss = criterion(output, target)
print("Loss:", loss.item())
# Backward pass
loss.backward()
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Update weights
optimizer.step()
# Zero gradients
optimizer.zero_grad()
# Save model
torch.save(model.state_dict(), "model.pth")
print("done")