import torch
import numpy as np
from dqn.dqn_model import DQN
from environment import SumoTrafficEnv

env = SumoTrafficEnv("sumo_env/traffic.sumocfg")
env.start()

model = DQN(env.state_size, env.action_size)
model.load_state_dict(torch.load("dqn_traffic_model.pth"))
model.eval()

state = torch.tensor(env.reset(), dtype=torch.float32)
done = False

while not done:
    action = torch.argmax(model(state)).item()
    next_state, reward, done = env.step(action)
    state = torch.tensor(next_state, dtype=torch.float32)

env.close()
print("Test completed!")
