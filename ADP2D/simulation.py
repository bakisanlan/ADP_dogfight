import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time
import matplotlib.pyplot as plt
from Train import *
from Plotter import plot_trajectories
from Timer import Timer


from env import AirCombatEnv

# Create environment
env = AirCombatEnv(boundary=60.0, velocity=2.5, 
                    max_bank_blue=np.radians(23),
                    max_bank_red=np.radians(18))

# Train Agent or Load Weights
gamma=0.95
# w_final = train(env, gamma = gamma)
#np.save('wfinal.npy', w_final)
w_final = np.load('weights_lookahead3_v2/w49.npy')
env.w = w_final
env.gamma = gamma


# Game Loop
_ = env.reset()
state = np.array([0 , 0 ,0, np.radians(5),
                  0, 3, 0, np.radians(-18)])
env.state = state
States = []
States.append(state)
done = False
cum_reward = 0.0
step_count = 0
step_final = 15
gamma = 0.95

while not done and step_count < 20:
    # choose blue_action = argmax_{a} [ r + gamma * w^T phi(s_next) ]

    J_best = -9999999
    n_lookahead = 3
    state_org = state
    
    for action in env.blue_actions:
        
        state_temp, _, _ = env.simulate_step(state_org, action)    
            
        for _ in range(n_lookahead):
        
            # Find nominal best blue action
            blue_nom_action = env._blue_nom_action(state_temp)
            state_temp, reward, _ = env.simulate_step(state_temp, blue_nom_action)
            
        J_current = reward + gamma * np.dot(w_final, env.feature_func(state_temp))
        print(J_current)
        
        if J_current > J_best:
            J_best = J_current
            best_blue_action = action

    # now actually step in the real environment
    
    # with Timer('3d'):
    best_red_action = env._red_3step_lookahead(state, best_blue_action)

    # with Timer('step'):
    state, reward, done = env.step(best_blue_action,best_red_action)
    States.append(state)

    cum_reward += reward
    step_count += 1

plot_trajectories(States)
print(f"Test episode finished in {step_count} steps, total reward {cum_reward:.2f}")

# with Timer('ded'):
#     for i in range(100000):
#         state, reward, done = env.step(1,1)
    
# print(i)
