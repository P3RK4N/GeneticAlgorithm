import os 
import torch

def load_tensor(dir, name):
    path = os.path.join(dir, name+'.txt')
    mat = []
    with open(path, 'r') as f:
        for line in f:
            mat.append([
                float(el) for el in line.strip().split()
            ])
    return torch.tensor(mat)

def load_body(dir):
    x = load_tensor(dir, 'points')
    dx = torch.zeros_like(x)
    bones = load_tensor(dir, 'bones').long()
    muscles = load_tensor(dir, 'muscles').long()
    rods = torch.cat((bones, muscles))
    rod_lengths = vlen(x[rods[:, 0]]-x[rods[:, 1]])
    rod_colors = torch.tensor([255]*len(bones) + [35]*len(muscles))
    return x, dx, rods, rod_lengths, rod_colors, len(muscles)

def make_random_body(n_points):
    x = torch.randint(0, 1000, (n_points, 2)).float()
    dx = torch.zeros_like(x)
    rods = torch.randint(low=0, high=len(x)-1, size=(len(x), 2))
    rod_lengths = torch.randint(low = 100, high = 200, size=(len(x), ))
    rod_colors = torch.randint(low=10, high=255, size=(len(x),))
    return x, dx, rods, rod_lengths, rod_colors, 0

def duplicate(x, n):
    return torch.cat([x for _ in range(n)])
def duplicate_and_add(x, n, to_add):
    return torch.cat([x + i*to_add for i in range(n)])

def load_batch_body(dir, batch_size):
    x, dx, rods, rod_lengths, rod_colors, n_muscles = load_body(dir)
    rods = duplicate_and_add(rods, batch_size, len(x))
    x, dx, rod_lengths, rod_colors = [
        duplicate(x, batch_size) 
        for x in [x, dx, rod_lengths, rod_colors]
    ]
    return x, dx, rods, rod_lengths, rod_colors, n_muscles

def vlen(v):
    return (v**2).sum(dim=1, keepdim=True).sqrt()

def norm(v, length):
    return v / (length + 1e-6)
