import math
import random
import time
import torch 

def matrix_mul(X, Y):
    result = [[0 for _ in Y[0]] for _ in X]
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    return result

def batch_matrix_mul(x, y):
    result = []
    for mx, my in zip(x, y):
        result.append(matrix_mul(mx, my))
    return torch.tensor(result)


class Nets:
    '''
    This is a container class that contains n Linear layers.
    It executes them one by one and feeds the output of one
    into the other.
    At the end output is clamped between `min_change` and `max_change`.
    '''
    def __init__(self, n_nets, arch, device, min_change=-0.4, max_change=0.4):
        '''
        Args:
            n_nets: int, number of networks
            arch: List[int]
                the dimensionality of every layer, so if the input is 32 dim,
                output 33 and we want to have 2 layers of 13 and 17 neurons,
                than this is [32, 13, 17, 33]
            device: torch.device, where to put the layers ('cpu' or 'cuda')
            min_change: int, maximum value by which muscle can contract
            max_change: int, maximum value by which muscle can expand
        '''
        self.layers = [
            Linear(n_nets, ni, no, device) for ni, no in zip(arch[:-1], arch[1:])
        ]
        self.n_nets = n_nets
        self.min_change = min_change
        self.max_change = max_change

    def __call__(self, x):
        for lay in self.layers[:-1]:
            x = torch.relu(lay(x))
        x = self.layers[-1](x)
        return torch.clamp(x, self.min_change, self.max_change)

    def mutate (self, std, keep_top=0.1, idxs=None):
    	# just call all layers to mutate themselves
        for lay in self.layers:
            lay.mutate(std, keep_top, idxs)
    
    def replace(self, fitness, bottom_perc, top_perc):
        '''
        higher fitness is better
        replace the bottom perc of nets with top perc of nets
        '''
        idxs = torch.argsort(fitness)
        for lay in self.layers:
            lay.replace(idxs, bottom_perc, top_perc)
        return idxs

class Linear:
    '''
    Contains two things:
    	self.lin is a batch of matrices, so if every matrix is of shape 32x33
    		than this would have shape (n, 32, 33)
    	self.bias is a batch of biases, so if every bias is of shape 32x1
    		than this would have shape (n, 32, 1)
    		
    	to get output you matrix multiply the input with self.lin and add self.bias
    	In the context of evolution batch_size would be population size
    '''
    def __init__(self, batch_size, infeatures, outfeatures, device):
        '''
        infeatures and outfeatures give matrix shape
        '''
        scale = math.sqrt(6 / infeatures) * 2 
        self.lin = (torch.rand(
                (batch_size, outfeatures, infeatures), 
                device=device, dtype=torch.float32
            ) - 0.5) * scale
        self.bias = (torch.rand(
                (batch_size, outfeatures, 1), 
                device=device, dtype=torch.float32
            ) - 0.5 ) * scale
        self.batch_size = batch_size

    def __call__(self, x):
        return batch_matrix_mul(self.lin, x) + self.bias

    def mutate(self, std, keep_top, idxs):
        # add gaussian noise to matrices, but keep `keep_top` ones without mutating
        keep_topnt = int(self.batch_size*(1-keep_top))
        idxs = idxs[:keep_topnt]
        # so inneficient... TODO
        self.lin[idxs] += torch.randn_like(self.lin[idxs]) * std
        self.bias[idxs] += torch.randn_like(self.bias[idxs]) * std

    def replace(self, idxs, bottom_perc, top_perc):
        '''
        sorts the matrices within the layer by fitness.
        it than replaces the bottom_perc of matrices by the top_perc
        '''
        # print(fitness[:10])
        # print(idxs[:10])
        topn = int(self.batch_size * top_perc)
        topnt = len(idxs) - topn
        botn = int(self.batch_size * bottom_perc)
        assert botn >= topn and botn%topn==0, (topn, botn)

        tops_lin = self.lin[idxs[topnt:]]
        tops_bias = self.bias[idxs[topnt:]]
        for thres in range(0, botn, topn):
            idxs2swap = idxs[thres:thres+topn]
            self.lin[ idxs2swap ] = tops_lin
            self.bias[idxs2swap] = tops_bias


# test if everything is working
if __name__ == '__main__':
    population_size = 1000
    n_generations = 100
    device = 'cpu'
    nets = Nets(population_size, [1, 3, 4, 1], device)

    for i in range(n_generations):
        outs = nets(torch.ones((population_size,1,1), dtype=torch.float32, device=device))
        outs = outs[:, 0, 0]
        nets.replace(outs, 0.3, 0.2)
        nets.mutate(std=1)
        print(torch.max(outs))
        # break

