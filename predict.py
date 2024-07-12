import torch
from torch.nn import ReLU


if __name__ == '__main__':
    tensor1 = torch.randint(-2, 2, (4, 4))
    tensor2 = torch.randint(0, 2, (4, 4))
    m = ReLU()
    print(tensor1)
    print(m(tensor1))