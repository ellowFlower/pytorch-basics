import torch
import numpy as np


"""
Initialize a tensor
"""
# from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy array
np_array = np.array(data)
x_numpy = torch.from_numpy(np_array)

# from another tensor
x_ones = torch.ones_like(x_data) # retains properties of x_data; values are all 1
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides data type of x_data; values are random from 0-1

# with random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


"""
Attributes of a Tensor
Tensor attributes describe their shape, datatype, and the device on which they are stored.
"""
tensor = torch.rand(3,4)
print(f"Shape of Tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(tensor)


"""
Operations on Tensors
"""
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[:, -1])
tensor[:, 1] = 0
print(tensor)
print('Jointet tensors: ', torch.cat([tensor, tensor, tensor], dim=1))

# matrix multiplication
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
y1 = tensor1 @ tensor2 
y2 = tensor1.matmul(tensor2) 
print(y1)
print(y2)

# elementwise product
z1 = tensor1 * tensor2
z2 = tensor1.mul(tensor2)
print(z1)

# single element tensor to numerical value
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in place operations: change parameter value; always have _ suffix
tensor.add_(5) # adds 5 every element in tensor































