import torch
from tqdm import tqdm 
from render import show_body as show
import sys
from utils import load_body, vlen, norm, make_random_body

N_POINS = 20
N_STEPS = 1000
GROUND = 200

GROUND_FORCE = 1
COEFF_FRICTION = 1
GRAVITY_STRENTH = 0.5
DAMPING_COEFF_GROUND = 1.

ROD_SPRINGINESS = 0.1
ROD_DAMPNESS = 0.1

device = 'cpu'


gravity_force = torch.tensor([[0, -GRAVITY_STRENTH]], device=device)
ground = torch.tensor([[1.0 * GROUND]], device=device)
ground_force_direction = torch.tensor([[0, GROUND_FORCE]], device=device) 

    
# simulation loop
def step(x, dx, rods, rod_lengths, nrods_in_body, population_size):
    force = torch.zeros_like(x)

    # add ground force
    ground_force = torch.relu((ground - x[:, 1:2]) * ground_force_direction)
    ground_force += torch.sigmoid(ground - x[:, 1:2]) * ground_force_direction * (-dx[:, 1:2]) * DAMPING_COEFF_GROUND
    force += ground_force
    
    # add gravity
    force += gravity_force

    # add rod force
    # get points at the ends of rods and their velocities
    ax, bx = x[rods[:, 0]], x[rods[:, 1]]
    adx, bdx = dx[rods[:, 0]], dx[rods[:, 1]]
    # get their position and velocity difference and their lens and norms
    ab, dab = ax - bx, adx - bdx
    len_ab = vlen(ab)
    len_dab = vlen(dab)
    norm_ab = norm(ab, len_ab)
    norm_dab = norm(dab, len_dab)
    # loop over them, calculate the force and add
    for i in range(nrods_in_body):
        rodlen = rod_lengths[i::nrods_in_body]
        len_abi = len_ab[i::nrods_in_body]
        norm_abi = norm_ab[i::nrods_in_body]
        len_dabi = len_dab[i::nrods_in_body]
        norm_dabi = norm_dab[i::nrods_in_body]

        f = (rodlen - len_abi) * norm_abi * ROD_SPRINGINESS        
        f -= len_dabi * norm_dabi * ROD_DAMPNESS

        iis = rods[i::nrods_in_body, 0]
        jjs = rods[i::nrods_in_body, 1]
        force[iis] += f
        force[jjs] -= f 

    # add friction
    # it has to happen last because it needs to know the expected velocity0
    expected_velocity = ev = dx + force
    len_ev = vlen(ev)
    len_friction = torch.minimum(
        vlen(ground_force * COEFF_FRICTION),
        len_ev
    )
    force -= len_friction * norm(ev, len_ev)
 
    # simulation step
    dx += force
    x += dx

    return x, dx


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dir = sys.argv[1]
        x, dx, rods, rod_lengths, rod_colors, _ = load_body(dir)
    else:
        x, dx, rods, rod_lengths, rod_colors, _ = make_random_body(N_POINS)

    for i in tqdm(range(N_STEPS)):
        x, dx = step(x, dx, rods, rod_lengths)
        show(x, rods, rod_colors)
