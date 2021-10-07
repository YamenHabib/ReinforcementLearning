import numpy as np
import itertools
from random import shuffle
from copy import deepcopy
# from tqdm.notebook import tqdm
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import *
from models import *

print(device)

features = get_example_features()
print(features.shape)
# plot_features(features.cpu().detach().numpy(), 4, 4)

MODEL_PATH = 'models/g.net'

def augment(batch):
    # random horizontal flip
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask] = batch['states'][flip_mask].flip(-1)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] > 0, 4 - batch['actions'][flip_mask], 0) # 1 -> 3, 3 -> 1

    # random vertical flip (and also diagonal)
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask]  = batch['states'][flip_mask].flip(-2)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] < 3, 2 - batch['actions'][flip_mask], 3) # 0 -> 2, 2 -> 0

    # shuffle opponents channels
    permuted_axs = list(itertools.permutations([0, 1, 2]))
    permutations = [torch.tensor(permuted_axs[i]) for i in np.random.randint(6, size=len(batch['states']))]
    for i, p in enumerate(permutations):
        shuffled_channels = torch.zeros(3, batch['states'].shape[2], batch['states'].shape[3])
        shuffled_channels[p] = batch['states'][i, 1:4]
        batch['states'][:, 1:4] = shuffled_channels
    return batch

def rollout(player, env, players, buffers, gammas, lambdas):
    rewards = []
    values  = []
    # shuffle players indices
    shuffle(players)
    trainer = env.train(players)
    observation = trainer.reset()
    prev_obs = observation
    done = False
    prev_heads = [None for _ in range(4)]
    # start rollout
    while not done:
        # cache previous state
        for i, g in enumerate(observation['geese']):
            if len(g) > 0:
                prev_heads[i] = prev_obs['geese'][i][0]
        prev_obs = observation
        # transform observation to state
        state = get_features(observation, env.configuration, prev_heads)
        # make a move
        action, logp, v = player.raw_outputs(state)
        # observe
        observation, reward, done, _ = trainer.step(['NORTH', 'EAST', 'SOUTH', 'WEST'][action])

        # data -> buffers
        buffers['states'].append(state)
        buffers['actions'].append(action)
        buffers['log-p'].append(logp.cpu().detach())
        # save rewards and values
        r = get_rewards(reward, observation, prev_obs, done)
        rewards.append(r)
        values.append(v)
    advs, rets = get_advantages_and_returns(rewards, values, gammas, lambdas)
    # add them to buffer
    buffers['adv'] += advs
    buffers['ret'] += rets

def runner(net, env, samples_threshold, gammas, lambdas):
    data_buffers = {'states': [],'actions': [], 'log-p': [], 'adv': [], 'ret': []}
    samples_collected = 0
    samples_bar = tqdm(total=samples_threshold, desc='Collecting Samples', leave=False)
    
    player = RLAgent(net, stochastic=True)
    opponents = [RLAgent(net, stochastic=False) for _ in range(3)]
    
    while True:
        rollout(player, env, players=[None] + opponents, buffers=data_buffers, gammas=gammas, lambdas=lambdas)
        samples_bar.update(len(data_buffers['states']) - samples_collected)
        samples_collected = len(data_buffers['states'])
        
        if samples_collected >= samples_threshold:
            samples_bar.close()
            return data_buffers

def train(net, optimizer, n_episodes=25, batch_size=256, samples_threshold=10000, n_ppo_epochs=25):
    gammas  = 0.8
    lambdas = 0.7  
    avg_loss = 0
    for episode in range(n_episodes): #tqdm(range(n_episodes), desc='Episode', leave=False):
        net.eval()        
        buffers = runner(net, env, samples_threshold, gammas=gammas, lambdas=lambdas)
        net.train()
        avg_loss = 0
        dataloader = DataLoader(GeeseDataset(buffers), batch_size=batch_size, shuffle=True, num_workers=2)
        for epoch in range(n_ppo_epochs):
            for batch in dataloader:
                losses = compute_losses(net, batch, c1=1, c_ent=0.01) #augment
                loss = losses['actor'] + losses['critic'] + losses['entropy']
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item()
        print(f"Episode: {episode}, Loss: {avg_loss/n_ppo_epochs}")
        torch.save(net.state_dict(), MODEL_PATH)


net = GNet().to(device)
optimizer = Adam(net.parameters(), lr=5e-6)
train(net, optimizer, n_episodes=50, batch_size=256, samples_threshold=10000, n_ppo_epochs=16)

# # torch.save(net.state_dict(), MODEL_PATH)

# # checking that all is working.
# net = GNet().to(device)
# net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# env = make("hungry_geese", debug=False)
# env.reset()
# net.eval()
# env.run([RLAgent(net, stochastic=False),
#          RLAgent(net, stochastic=False),
#          RLAgent(net, stochastic=False),
#          RLAgent(net, stochastic=False)])
# env.render(mode='ipython', width=500, height=400)