'''2
_
_
.'''
import torch
import torch.optim as optim
from torch.distributions import Categorical
from gridworld_env import GridWorldEnv
from actor_model import ActorModel

env = GridWorldEnv()
model = ActorModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        state_tensor = torch.tensor([state[0] * env.size + state[1]], dtype=torch.float32)
        probabilities = model(state_tensor)
        m = Categorical(probabilities)
        action = m.sample()
        state, reward, done, _ = env.step(action.item())
        log_prob = m.log_prob(action)
        log_probs.append(log_prob)
        rewards.append(reward)

    # Policy gradient update
    discounts = [0.99 ** i for i in range(len(rewards))]
    R = sum([a * b for a, b in zip(discounts, rewards)])
    policy_loss = [-log_prob * R for log_prob in log_probs]
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
