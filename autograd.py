import torch


x = torch.ones(5)
y = torch.zeros(3)
w = torch.rand(5, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


"""
Computing Gradients
compute the derivatives of our loss function with respect to parameters
"""
loss.backward()
print(w.grad)
print(b.grad)























