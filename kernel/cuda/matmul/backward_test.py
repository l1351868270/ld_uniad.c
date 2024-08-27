import torch 
from torch import nn
import torch.nn.functional as F
nn.CrossEntropyLoss()
torch.random.manual_seed(0)
# # Example of target with class indices
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss = F.cross_entropy(input, target)
# loss.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
loss = F.cross_entropy(input, target)
print(loss)
print(loss.grad_fn.name)
print(input.grad_fn)
loss.backward()

x = torch.ones(2, 2, requires_grad=True)
y = x ** 2
print(y.grad_fn)