import torch
import torch.nn as nn
import torch.optim as optim

# Generate a synthetic dataset: y = 3x + noise
X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3 * X + 0.5 * torch.randn(X.size())

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
for epoch in range(200):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Trained weight: {model.linear.weight.item():.3f}, bias: {model.linear.bias.item():.3f}')
