import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time
import matplotlib.pyplot as plt
from Train3d import *
from Plotter import plot_trajectories
from Timer import Timer


from env3d import AirCombatEnv

# Create environment
env = AirCombatEnv(velocity=2.5)

# Train Agent or Load Weights
gamma=0.95
# w_final = train(env, gamma = gamma)  #For training uncomment it

w_final = np.load('weights/weights_3d_v3/w19.npy')
env.w = w_final
env.gamma = gamma



# Game Loop
# For initial position of two aircraft, below different initial states should be selected with uncommenting one of them

##Random initial state
# state = env.reset()

#Flight 1
state = np.array([0 ,  0 ,0, 0 ,0,
                  3.5 ,0, 0,0,0] )

##Flight 2
# state = np.array([0 , 3 ,0,-np.radians(23) ,0,
#                   0 ,0, 0,np.radians(-18),0] )
##Flight 3
# state = np.array([3.5, 0 ,0,0 ,0,
#                   0 ,0, 0,0,0] )

##Flight 4
# state = np.array([4, 0 ,0,0 ,0,
#                   0 ,0, 0,-np.pi,0] )

##Flight 5
# state = np.array([0, 0 ,3,np.pi/4 ,0,
#                   -0.25 ,0.25, 0,-np.pi/4,0] )

##Flight 6
# state = np.array([-0.1, 0 ,0,np.pi*1.5 ,0,
#                   0.1 ,0, 0, np.pi/2,0] )

env.state = state
States = []
States.append(state)
done = False
cum_reward = 0.0
step_count = 0
step_final = 15
gamma = 0.95

while (not done) and (step_count < 30):
    # choose blue_action = argmax_{a} [ r + gamma * w^T phi(s_next) ]

    J_best = -9999999
    n_lookahead = 3
    state_org = state
    
    red_action = env._red_minimax(state_org,1)    
    for action in env.blue_actions:
        state_temp, _, _ = env.step_outer(state_org,action,red_action)   
        
        for _ in range(n_lookahead):
        
            # Find nominal best blue action
            blue_nom_action = env.blue_nom_action(state_temp)
            red_action = env._red_minimax(state_temp,1)    
            state_temp, reward, _ = env.step_outer(state_temp, blue_nom_action, red_action)
            
        J_current = reward + gamma * np.dot(w_final, env.feature_func(state_temp))
        # print(J_current)
        # print(step_count)
        
        if J_current > J_best:
            J_best = J_current
            best_blue_action = action
    
    # # now actually step in the real environment
    
    best_red_action = env._red_minimax(state)
    # with Timer('step'):
    
    # # # ##minimax_search, for seeing what would happen if both of them use same policy minimax search without learning uncomment below to line
    # best_blue_action = env._blue_minimax(state)
    # best_red_action = env._red_minimax(state)
    
    state, reward, done = env.step(best_blue_action,best_red_action)
    States.append(state)
    print(best_blue_action,best_red_action)

    cum_reward += reward
    step_count += 1


plot_trajectories(States,1)
