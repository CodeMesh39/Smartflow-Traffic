import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn.dqn_model import DQN
from environment import SumoTrafficEnv

env = SumoTrafficEnv("sumo_env/traffic.sumocfg")
env.start()

model = DQN(env.state_size, env.action_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

epsilon = 1.0
gamma = 0.9
episodes = 30
episode_rewards = []

for episode in range(episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32)  # Reset state here
    total_reward = 0

    for step in range(200):
        if np.random.rand() < epsilon:
            action = np.random.randint(env.action_size)
        else:
            action = torch.argmax(model(state)).item()

        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        target = reward + gamma * torch.max(model(next_state)).item()
        prediction = model(state)[action]

        loss = criterion(prediction, torch.tensor(target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon *= 0.95
    episode_rewards.append(total_reward)
    print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

env.close()

# Plot reward graph
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward vs Episodes (DRL Traffic Signal)")
plt.grid()
plt.show()

# Save the model after training
torch.save(model.state_dict(), "dqn_traffic_model.pth")
print("Model saved successfully!")
