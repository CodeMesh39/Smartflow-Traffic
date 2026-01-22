import traci
import numpy as np

class SumoTrafficEnv:
    def __init__(self, sumo_cfg):
        self.sumo_cfg = sumo_cfg
        self.state_size = 4
        self.action_size = 2  # keep or switch

    def start(self):
        traci.start([
            "sumo",
            "-c", self.sumo_cfg,
            "--start"
        ])

    def reset(self):
        return self.get_state()

    def get_state(self):
        lanes = traci.trafficlight.getControlledLanes("c")[:4]
        queues = [
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in lanes
        ]
        return np.array(queues, dtype=np.float32)

    def step(self, action):
        if action == 1:
            phase = traci.trafficlight.getPhase("c")
            traci.trafficlight.setPhase("c", (phase + 1) % 4)

        traci.simulationStep()

        next_state = self.get_state()

        lanes = traci.trafficlight.getControlledLanes("c")[:4]
        waiting_time = sum(
            traci.lane.getWaitingTime(lane) for lane in lanes
        )

        reward = -(0.7 * np.sum(next_state) + 0.3 * waiting_time)

        done = traci.simulation.getTime() > 10000
        return next_state, reward, done

    def close(self):
        traci.close()
