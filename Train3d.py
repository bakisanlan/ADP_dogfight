import matplotlib.pyplot as plt
from typing import Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
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
    # np.random.seed(seed)
    
    # 3.1 Prepare feature dimension
    # We'll get dimension by extracting from a dummy state
    dummy_s = env.reset(0,0,0,0,0,0,0,0,0,0)
    phi_dim = len(env.feature_func(dummy_s))
    w = np.zeros(phi_dim, dtype=np.float32)
    
    # 3.2 Sample states
    States = env.collectSample(num_samples)
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
                
                red_action = env._red_minimax(state)
                state_next, reward, done = env.step_outer(state,blue_action,red_action)

                if i != 0: 
                    val_next = 0.0 if done else np.dot(w, env.feature_func(state_next,state))

                else:
                    val_next = 0.0 if done else env.S_function(state_next)
                    
                q_val = reward + gamma * val_next
                if q_val > best_val:
                    best_val = q_val
                    StatesNext[j] = state_next
                    
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
        np.save('weights_3d_v1/'+ 'w'+ str(i), w_new)

        
        # Track error
        pred = X.dot(w)
        mse = np.mean((pred - y)**2)
        print(f"Iteration {i+1}/{num_iterations}, MSE={mse:.4f}")
    
    return w

# -----------------------------
# 4. Main / Example Usage
# -----------------------------
def train(env, gamma=0.95,num_samples=1000,num_iterations=50,seed=45):
        
    # Run approximate dynamic programming
    w_final = approximate_value_iteration(env,gamma,num_samples,num_iterations,seed)
    
    print("Learned weights shape:", w_final.shape)
    print("Example of learned weights:", w_final[:10], "...")
    
    return w_final

