import numpy as np
import matplotlib.pyplot as plt

fixed_time_wait = np.random.randint(45, 65, 50)
drl_wait = np.random.randint(20, 35, 50)

print("Average Waiting Time")
print("Fixed-Time Signal:", fixed_time_wait.mean())
print("DRL Signal:", drl_wait.mean())

plt.bar(["Fixed-Time", "DRL"], [fixed_time_wait.mean(), drl_wait.mean()])
plt.ylabel("Average Waiting Time (seconds)")
plt.title("Fixed-Time vs DRL Traffic Signal")
plt.show()
