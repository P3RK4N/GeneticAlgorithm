import os
import sys
import torch
import pickle
import time
from nn import Nets
from tqdm import tqdm
from PIL import Image
from simulation import step, device
# from render import show_body
from utils import load_batch_body

POPULATION_SIZE = 1
N_STEPS_MIN = 350
N_STEPS_MAX = 350
N_GENERATIONS = 1000
SHOW_FIRST = 50
VISUALIZE_EVERY_N_GENERATIONS = 50

x, dx, rods, rod_lengths, rod_colors, n_muscles = load_batch_body(sys.argv[1], POPULATION_SIZE)
input_size = (len(x) + len(dx)) * 2 // POPULATION_SIZE + 2*2
# we give it number of nets that need to be instantiated and neural net architecture
nets = Nets(POPULATION_SIZE, [input_size, 32, 32, n_muscles], device)

# this is used to get first n creatures to visualize them
# cause visualizing 1k is to much
def crop_first(x, n=SHOW_FIRST):
    return x[:n*len(x)//POPULATION_SIZE]
def get_best(x, sub=False):
    i = torch.argmax(fitness)
    res =x[len(x)//POPULATION_SIZE*i:len(x)//POPULATION_SIZE*(i+1)]
    if sub:
        return res - torch.min(res)
    return res

population_imgs = []
best_one_imgs = []
best = 0
fitness = torch.zeros_like(x)
for igen in range(N_GENERATIONS):
    x, dx, _, _, _, _ = load_batch_body(sys.argv[1], POPULATION_SIZE)
    x = x.to(device)
    dx = dx.to(device)
    rods = rods.to(device)
    rod_lengths = rod_lengths.to(device)
    n_points_in_body = len(x) // POPULATION_SIZE
    n_rods_in_body = len(rods) // POPULATION_SIZE
    
    # we linearly increase number of steps in every episode
    alpha = igen / N_GENERATIONS
    n_steps = alpha * N_STEPS_MAX + (1-alpha) * N_STEPS_MIN

    energy = 0
    tep = time.time()
    for istep in tqdm(range(int(n_steps))):

        # the network should generalize easier if the points are in relative form
        # so we take the fist point of the body and find the positions and velocities 
        # of other points in the body relative to it
        reference_points     =  x[::n_points_in_body]
        reference_velocities = dx[::n_points_in_body]
        # to subtract reference_points from x they have to be of the same dimension
        # we have to repeat reference_points so that there are the same number of them
        # and the points
        ireference_points     =  torch.repeat_interleave(reference_points, n_points_in_body, dim=0)
        ireference_velocities =  torch.repeat_interleave(reference_velocities, n_points_in_body, dim=0)
        relative_x  = x - ireference_points        
        relative_dx = dx - ireference_velocities 
        # merge relative positions, velocities, and the absolute position and velocity
        # of the reference (first) body point to one big tensor
        obervations = torch.cat((
            relative_x.view(POPULATION_SIZE, -1),
            relative_dx.view(POPULATION_SIZE, -1),   
            reference_points,
            reference_velocities,
        ), dim=1)

	    # send the observations to neural nets and get their predictions
        t = time.time()
        # print(obervations.device)
        preds = nets(obervations[:, :, None]).view(POPULATION_SIZE, -1)
        # print('nettime', time.time()-t)
        energy += (preds**2).sum(dim=1)

        # the neural net predicts by how much the muscles should contract or expand
        # to calculate the final length we need to add the base length and the predicted
        # change, to do that we need to pad the predicted change with zeros since 
        # it includes only the muscles
        drod_lengths = torch.cat((
            torch.zeros((POPULATION_SIZE, len(rods)//POPULATION_SIZE - n_muscles), device=device),
            preds
        ), dim=1)
        wanted_rod_lengths = rod_lengths.view(POPULATION_SIZE, -1) * (1+drod_lengths)
        wanted_rod_lengths = wanted_rod_lengths.view(-1, 1)

        # physics step and visualization
        t = time.time()
        x, dx = step(x, dx, rods, wanted_rod_lengths, n_rods_in_body, POPULATION_SIZE) 

        # print('phytime', time.time()-t)
        if igen % VISUALIZE_EVERY_N_GENERATIONS == 0:
            #img = show_body(crop_first(x), crop_first(rods), crop_first(rod_colors), text="generation = "+str(igen))
            #population_imgs.append(img)
            # visualize only the first body
            # img = show_body(get_best(x), get_best(rods, True), get_best(rod_colors), text="generation = "+str(igen))
            # best_one_imgs.append(img)
            pass
    tepend = time.time()


    # print(preds[:3])
    x = x.view(POPULATION_SIZE, -1, 2)
    fitness = x[:, :, 0].mean(dim=1)
    # fitness -=  0.3 * energy
    # print(fitness[:10])
    print(igen, "maxf=", torch.max(fitness).item(), "time per ep=", tepend-tep)
    idxs = nets.replace(fitness, 0.5, 0.1)
    nets.mutate(std=1, keep_top=0.1, idxs=idxs)
    if torch.max(fitness) > 2000:
        0/0
    0/0



# this is to save everything after training
population_imgs = [Image.fromarray(img) for img in population_imgs]
best_one_imgs   = [Image.fromarray(img) for img in best_one_imgs]
if len(population_imgs) > 0:
    population_imgs[0].save(
        os.path.join(sys.argv[1], 'population.gif'), 
        save_all=True, append_images=population_imgs[1:], duration=80, loop=0)
if len(best_one_imgs) > 0:
    best_one_imgs[0].save(
        os.path.join(sys.argv[1], 'best_one.gif'), 
        save_all=True, append_images=best_one_imgs[1:], duration=80, loop=0)
with open(os.path.join(sys.argv[1], 'model.pckl'), 'wb') as f:
    pickle.dump(nets, f)

