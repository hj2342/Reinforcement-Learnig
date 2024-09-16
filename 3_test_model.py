"""Test the trained policy (for REINFORCE).

Author: Elie KADOCHE.
"""

import torch
import numpy as np

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    policy = ActorModelV0()
    actor_path = "/Users/hariharanjanardhanan/Desktop/mlpr/saved_models/actor_0.pt"

    # ------------------------------------------
    # ---> TODO: UNCOMMENT FOR SECTION 4 ONLY
    env = CartpoleEnvV1()
    policy = ActorModelV1()
    actor_path = "./saved_models/actor_1.pt"
    # ------------------------------------------

    # Testing mode
    # policy.eval()
    # print(policy)

       # Load the trained policy weights into the policy object
    # state_dict = torch.load(actor_path, map_location=torch.device("cpu"))
    # policy.load_state_dict(state_dict)
    policy.eval()  # Set the model to evaluation mode
    print(policy)
    policy=torch.load(actor_path)
    # Reset environment
    state, _ = env.reset(seed=None)
    # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert state to tensor and add batch dimension

    total_reward = 0.0
    terminated = False
    while not terminated:
        probabilities = policy(state)


       # ---> TODO: how to select an action
        action = torch.argmax(probabilities).item()


        # One step forward
        state, reward, terminated, _, _ = env.step(action)
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Update state tensor

        # Render (or not) the environment
        total_reward += reward
        env.render()  # Ensure render is implemented or handled in CartpoleEnvV0/V1

    # Print the total reward
    print(f"Total reward = {total_reward}")