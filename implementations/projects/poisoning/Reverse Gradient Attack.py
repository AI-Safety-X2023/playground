import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model and optimizer
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Function to compute gradients
def compute_gradient(model, data, target):
    """Computes gradient for a given data point."""
    data.requires_grad = True  # Ensure trackable gradients
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return data.grad.clone()

# Malicious gradient
def generate_malicious_gradient(model, data, target, attack_type="gradient_ascent"):
    gradient = compute_gradient(model, data, target)
    if attack_type == "gradient_ascent":
        return -gradient  
    elif attack_type == "orthogonal":
        return gradient - (gradient @ gradient.T) * gradient 
    return gradient

# Invert the gradient
def invert_gradient(model, target_gradient, steps=1000, lr=0.1):
    poisoned_data = torch.randn_like(target_gradient, requires_grad=True)  
    optimizer = optim.Adam([poisoned_data], lr=lr)
    
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        poisoned_gradient = compute_gradient(model, poisoned_data, torch.randint(0, 10, (1,)))
        
        # Ensure loss is differentiable
        loss = ((poisoned_gradient - target_gradient) ** 2).sum()
        loss.backward()
        optimizer.step()
        
        history.append(poisoned_data.detach().clone())

    return poisoned_data.detach(), history

# Example run
for data, target in train_loader:
    data, target = data[0].unsqueeze(0), target[0].unsqueeze(0)  # Ensure batch shape
    target = target.long()  # Ensure correct dtype
    data.requires_grad = True

    target_gradient = generate_malicious_gradient(model, data, target, "gradient_ascent")
    target_gradient.requires_grad = True  # Ensure gradient tracking

    poisoned_data, history = invert_gradient(model, target_gradient)
    break  # Stop after one sample

# Visualization of gradient inversion
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    if i * 200 < len(history):
        ax.imshow(history[i * 200].view(28, 28).detach().numpy(), cmap='gray')
        ax.set_title(f"Step {i * 200}")
        ax.axis("off")
plt.show()
