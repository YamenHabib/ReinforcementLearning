import numpy as np
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns


import pickle
import bz2
import base64

# hungry-geese imports

from kaggle_environments import make, evaluate
env = make("hungry_geese", debug=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

def ids2locations(ids, prev_head, step, rows, columns):
    state = np.zeros((5, rows * columns))
    if len(ids) == 0:
        return state
    state[0, ids[0]] = 1 # goose head
    if len(ids) > 1:
        state[1, ids[1:-1]] = 1 # goose body
        state[2, ids[-1:]] = 1   # goose tail
        state[3, ids[:]] = 1       # whole body
    if step != 0:
        state[4, prev_head] = 1 # goose head one step before
  
    return state

def get_features(observation, config, prev_heads):
    rows, columns = config['rows'], config['columns']
    geese = observation['geese']
    index = observation['index']
    step = observation['step']
    
    # convert indices to locations
    locations = np.zeros((len(geese), 5, rows * columns))
    for i, g in enumerate(geese):
        locations[i] = ids2locations(g, prev_heads[i], step, rows, columns)
    
    if index != 0: # swap rows for player locations to be in first channel
        locations[[0, index]] = locations[[index, 0]]
    
    # put locations into features
    features = np.zeros((17, rows * columns))
    for k in range(4):
        features[k]     = locations[k][0]             # head
        features[k + 4] = locations[k][3]  # head + body + tail.
        features[k + 8] = locations[k][4]             # prev head
    
    features[-5] = np.sum(locations[:,3], 0)                                  # the whole bodies of all geese
    features[-4, observation['food']] = 1                                     # food channel
    features[-3, :] = (step % config['hunger_rate']) / config['hunger_rate']  # hunger danger channel
    features[-2, :] = step / config['episodeSteps']                           # timesteps channel
    features[-1, :] = float((step + 1) % config['hunger_rate'] == 0)          # hunger milestone indicator
    features = torch.Tensor(features).reshape(-1, rows, columns)
    # roll
    head_id = geese[index][0]
    head_row = head_id // columns
    head_col = head_id % columns
    features = torch.roll(features, ((rows // 2) - head_row, (columns // 2) - head_col), dims=(-2, -1))
    return features

def plot_features(features, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(rows):
        for j in range(cols):
            sns.heatmap(features[i * 4 + j], ax=axs[i, j], cmap='Blues',
                        vmin=0, vmax=1, linewidth=2, linecolor='black', cbar=False)
    plt.show()

def get_example_features():
    observation = {}
    observation['step'] = 104
    observation['index'] = 0
    observation['geese'] = [[46, 47, 36, 37, 48, 59, 58, 69],
                            [5, 71, 72, 6, 7, 73, 62, 61, 50, 51, 52, 63, 64, 53, 54],
                            [12, 11, 21, 20, 19, 8, 74, 75, 76, 65, 55, 56, 67, 1],
                            [23, 22, 32, 31, 30, 29, 28, 17, 16, 27, 26, 15, 14, 13, 24]]
    observation['food'] = [45, 66]
    prev_heads = [47, 71, 11, 22]
    return get_features(observation, env.configuration, prev_heads)


# 1 is the best, 4 is the worst
def get_rank(obs, prev_obs):
    geese, index = obs['geese'], obs['index']
    player_len = len(geese[index])
    survivors = [i for i in range(len(geese)) if len(geese[i]) > 0]
    if index in survivors: # if our player survived in the end, its rank is given by its length in the last state
        return sum(len(x) >= player_len for x in geese) # 1 is the best, 4 is the worst
    # if our player is dead, consider lengths in penultimate state
    geese, index = prev_obs['geese'], prev_obs['index']
    player_len = len(geese[index])
    rank_among_lost = sum(len(x) >= player_len for i, x in enumerate(geese) if i not in survivors)
    return rank_among_lost + len(survivors)

def get_rewards(env_reward, obs, prev_obs, done):
    geese = prev_obs['geese']
    index = prev_obs['index']
    step  = prev_obs['step']
    if done:
        rank = get_rank(obs, prev_obs)
        r1 = (1, -0.25, -0.75, -1)[rank - 1]
        died_from_hunger = ((step + 1) % 40 == 0) and (len(geese[index]) == 1)
        r2 = -1 if died_from_hunger else 0 # huge penalty for dying from hunger and huge award for the win
    else:
        if step == 0:
            env_reward -= 1 # somehow initial step is a special case
        r1 = 0
        r2 = max(0.1 * (env_reward - 1), 0) # food reward
    return r1 + r2

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# count_parameters(net)

def inv_discount_cumsum(array, discount_factor):
    res = [array[-1]]
    for x in torch.flip(array, dims=[0])[1:]:
        res.append(discount_factor * res[-1] + x)
    return torch.flip(torch.stack(res), dims=[0])

def get_advantages_and_returns(rewards, values, gamma, lam):
    # lists -> tensors
    rewards = torch.tensor(rewards, dtype=torch.float)
    values = torch.tensor(values + [0.]) # Baseline estimate is the output of value network  Vθ(s) .
    # calculate deltas, A and R
    deltas = rewards + gamma * values[1:] - values[:-1]
    advs = inv_discount_cumsum(deltas, gamma * lam).cpu().detach().tolist()
    rets = inv_discount_cumsum(rewards, gamma).cpu().detach().tolist()
    return advs, rets

def compute_losses(net,  data, c1, c_ent, clip_ratio=0.2):
    # move data to GPU
    states = data['states'].to(device)
    actions = data['actions'].to(device)
    logp_old = data['log-p'].to(device)
    returns = data[f'ret'].float().to(device)
    advs  = data['adv'].float().to(device)   
    
    # get network outputs
    logp_dist, values = net(states)
    logp = torch.stack([lp[a] for lp, a in zip(logp_dist, actions)])

    # compute actor loss
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advs
    actor_loss = -(torch.min(ratio * advs, clip_adv)).mean()
    
    # critic losses
    critic_loss= ((values - returns) ** 2).mean()
    
    # entropy loss
    entropy = Categorical(probs=torch.exp(logp_dist)).entropy()
    entropy[entropy != entropy] = torch.tensor(0.).to(device) # remove NaNs if any
    entropy_loss = -entropy.mean()
    
    return {'actor': actor_loss,
            'critic': (c1 * critic_loss),
            'entropy': c_ent * entropy_loss}
