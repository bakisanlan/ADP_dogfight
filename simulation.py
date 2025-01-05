import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import time
import matplotlib.pyplot as plt


from env import AirCombatEnv

def approximate_value_iteration(env, 
                                gamma=0.95,
                                num_samples=100000,
                                num_iterations=40,
                                seed=42):
    """
    Fitted Value Iteration using second-order polynomial features, 
    with 40 iterations on 1e5 states.
    
    1) Sample states (S_i).
    2) For each iteration, do:
        - For each S_i, compute Bellman backup:
          V_target(S_i) = max_{blue_action} [ r(S_i, blue_action) + gamma * w^T phi(S') ]
            where S' depends also on Red's 3-step lookahead.
        - Solve w = argmin_{w} sum ( w^T phi(S_i) - V_target(S_i) )^2
    3) Return final weights.
    """
    np.random.seed(seed)
    
    # 3.1 Prepare feature dimension
    # We'll get dimension by extracting from a dummy state
    dummy_s = env.reset(0,0,0,0,0,0,0,0)
    phi_dim = len(env.feature_func(dummy_s))
    w = np.zeros(phi_dim, dtype=np.float32)
    
    # 3.2 Sample states
    States = []
    for _ in range(num_samples):
        state = env.reset()  # random
        States.append(state)
    States = np.array(States, dtype=np.float32)
    StatesNext = States.copy()
    StatesPrev = States.copy()
    

    for i in range(num_iterations):
        X = []
        y = []
        
        for j, state in enumerate(States):
            
            best_val = -1e9
            for blue_action in env.blue_actions:
                # simulate one step from s
                
                state_next, reward, done = simulate_step(env, state, blue_action)

                if i != 0: 
                    val_next = 0.0 if done else np.dot(w, env.feature_func(state_next,state))

                else:
                    val_next = 0.0 if done else env.S_function(state_next)
                    
                q_val = reward + gamma * val_next
                if q_val > best_val:
                    best_val = q_val
                    
                    StatesNext[j] = state_next
                    a = 3
                    
            X.append(env.feature_func(state,StatesPrev[j]))
            y.append(best_val)
            
        StatesPrev = States.copy()        # t                   
        States     = StatesNext.copy()    # t+1

        X = np.array(X)
        y = np.array(y)
        
        # Solve least squares
        # w = (X^T X)^{-1} X^T y, or using np.linalg.lstsq
        w_new, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        w = w_new
        np.save('weights/'+ 'w'+ str(i) + 'npy', w_new)

        
        # Track error
        pred = X.dot(w)
        mse = np.mean((pred - y)**2)
        print(f"Iteration {i+1}/{num_iterations}, MSE={mse:.4f}")
    
    return w


def simulate_step(env, state, blue_action):
    """
    A helper function that simulates one step from state s 
    if Blue picks 'blue_action' and Red picks the best 3-step lookahead (approx).
    Return: next_state, immediate_reward, done_flag
    """
    red_best_action = env._red_3step_lookahead(state, blue_action)
    
    state_next, reward, done = env.step_outer(state,blue_action,red_best_action)
        
    return state_next, reward, done

# -----------------------------
# 4. Main / Example Usage
# -----------------------------
def train(env, gamma=0.95,num_samples=10000,num_iterations=40,seed=42):
        
    # Run approximate dynamic programming
    w_final = approximate_value_iteration(env,gamma,num_samples,num_iterations,seed)
    
    print("Learned weights shape:", w_final.shape)
    print("Example of learned weights:", w_final[:10], "...")
    
    return w_final


# Create environment
env = AirCombatEnv(boundary=60.0, velocity=2.5, 
                    max_bank_blue=np.radians(23),
                    max_bank_red=np.radians(18))

# Train Agent or Load Weights
#w_final = train(env)
#np.save('wfinal.npy', w_final)
w_final = np.load('weights/w39.npy')

# Game Loop
_ = env.reset()
state = np.array([0 , 0 , 0, 0,
                  3.5, 0, 0, 0])
env.state = state
States = []
States.append(state)
done = False
cum_reward = 0.0
step_count = 0
gamma = 0.95

while not done and step_count < 30:
    # choose blue_action = argmax_{a} [ r + gamma * w^T phi(s_next) ]
    best_blue_action = None
    best_val = -1e9
    for action in env.blue_actions:

        state_next, reward, done = simulate_step(env, state, action)

        val_next = 0.0 if done else np.dot(w_final, env.feature_func(state_next))

        q_val = reward + gamma * val_next
        if q_val > best_val:
            best_val = q_val
            best_blue_action = action
    
    # now actually step in the real environment
    best_red_action = env._red_3step_lookahead(state, best_blue_action)
    state, reward, done = env.step(best_blue_action,best_red_action)
    States.append(state)

    cum_reward += reward
    step_count += 1

def plot_trajectories(states):
    """
    Given a list of states, each of length >= 6:
      states[t] = [xB, yB, headingB, bankB, xR, yR, headingR, bankR]
    plots the Blue and Red positions over time.
    """
    # Extract x,y for Blue and Red across all time steps
    blue_x = [s[0] for s in states]  # xB
    blue_y = [s[1] for s in states]  # yB
    red_x  = [s[4] for s in states]  # xR
    red_y  = [s[5] for s in states]  # yR

    # Create the figure
    plt.figure(figsize=(6, 6))
    
    # Plot Blue trajectory
    plt.plot(blue_x, blue_y, '-o', color='blue', label='Blue Trajectory')
    # Mark the first point with a star
    plt.plot(blue_x[0], blue_y[0], 'b*', markersize=10, label='Blue Start')
    
    # Plot Red trajectory
    plt.plot(red_x, red_y, '-o', color='red', label='Red Trajectory')
    # Mark the first point with a star
    plt.plot(red_x[0], red_y[0], 'r*', markersize=10, label='Red Start')
    
    # Labeling and grid
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Blue vs. Red Position History")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # So that circles, angles, etc. look correct.
    plt.show()


plot_trajectories(States)
print(f"Test episode finished in {step_count} steps, total reward {cum_reward:.2f}")