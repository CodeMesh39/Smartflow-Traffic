# Adaptive Smart Traffic Signal Optimization Using Deep Reinforcement Learning

This project implements an AI-based adaptive traffic signal control system using Deep Reinforcement Learning (DQN) and SUMO simulation.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CodeMesh39/Smartflow-Traffic/blob/main/run_colab.ipynb)

## Features
- Real-time traffic simulation using SUMO
- Python–SUMO integration via TraCI
- Deep Reinforcement Learning (DQN)
- Adaptive signal timing
- Reward-based learning
- Performance comparison with fixed-time signals

## Technologies
- Python
- PyTorch
- SUMO
- TraCI
- NumPy
- Matplotlib

## How to Run
1. Install SUMO and add to PATH
2. Install dependencies:
   pip install torch numpy matplotlib traci sumolib
3. Run training:
   python train.py
4. Compare performance:
   python compare.py

## Results
- Reduced waiting time
- Improved traffic flow
- Adaptive real-time control

## Future Scope
- Emergency vehicle priority
- Multi-intersection control
- Real camera data
- Web dashboard
