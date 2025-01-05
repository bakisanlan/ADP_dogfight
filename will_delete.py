import time
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

States = []
for _ in range(10):
    state = env.reset()  # random
    States.append(state)
States = np.array(States, dtype=np.float32)
StatesNext = States.copy()
StatesPrev = States.copy()

print(StatesPrev[0])

StatesPrev[0] = np.array([1,2,3,4,5,6,7,8])

print(StatesPrev[0])
print(StatesNext[0])

import numpy as np

# Example array
array = np.array([1, 2, 3, 4, 5])

# Save array
np.save('weights/' + str(3) + '1', array)

# # Load array
# loaded_array = np.load('array.npy')
# print(loaded_array)


def simulate_step(env, state, blue_action):
    """
    A helper function that simulates one step from state s 
    if Blue picks 'blue_action' and Red picks the best 3-step lookahead (approx).
    We'll replicate a bit of env logic, ignoring environment done/boundary resets 
    except for the immediate step.
    Return: next_state, immediate_reward, done_flag
    """
    red_best_action = env._red_3step_lookahead(state, blue_action)
    
    state_next, reward, done = env.step_outer(state,blue_action,red_best_action)
        
    return state_next, reward, done

start_time = time.time()
for i in range(1000):
    state_next, reward, done = simulate_step(env, state, 0)
end_time = time.time()
print(f"simulate_step: {end_time - start_time:.5f} seconds")


w = np.random.rand(104)

start_time = time.time()
for i in range(1000):
    val_next = 0.0 if done else np.dot(w, env.feature_func(state_next,state))
end_time = time.time()
print(f"feature_func: {end_time - start_time:.5f} seconds")
