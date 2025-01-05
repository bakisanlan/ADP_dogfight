from env import AirCombatEnv
import numpy as np
env = AirCombatEnv()

States = []
for _ in range(10):
    state = env.reset()  # random
    States.append(state)
States = np.array(States, dtype=np.float32)

#print(States[-1])

# next_state, reward, done = env.step(0,0)

# print(next_state)

state = np.array([1 , 2 , np.pi/3 ,  np.pi/20, 
                  4 , 6 , -np.pi/6 , -np.pi/15])

# next_state, reward, done = env.step_outer(state,0,2)


red_best_action = env._red_3step_lookahead(state,1)



# print(reward)

# next_state, reward, done = env.step_outer(next_state,1,1)

# print(reward)

#print(env.ClosureAngle_find(state))

# second_order_no_bias = env.extract_features_13(next_state, state)
# print(second_order_no_bias)
#print(env.ClosureAngle_find(state))

