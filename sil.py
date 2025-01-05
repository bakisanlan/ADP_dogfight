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
