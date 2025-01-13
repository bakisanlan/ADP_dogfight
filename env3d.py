import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from Timer import Timer

# -----------------------------
# 1. Environment Definition
# -----------------------------
class AirCombatEnv:
    """
    A simplified 3D air combat environment with:
      - 2 aircraft (Blue, Red)
      - State includes positions (x,y), headings, bank angles, etc.
      - Blue controls: {LEFT, STRAIGHT, RIGHT} [discrete]
      - Red does a 3-step lookahead trying to maximize S, 
        while Blue tries to minimize S.
    """
    def __init__(self, 
                 boundary=60.0,
                 velocity=10,
                 max_bank_blue =np.radians(23),
                 max_bank_red  =np.radians(18)
                 ):
        self.boundary = boundary
        self.velocity = velocity
        self.max_bank_blue= max_bank_blue
        self.max_bank_red = max_bank_red
        self.max_flpath_blue = np.radians(30)
        self.max_flpath_red  = np.radians(25)
        self.max_nz_blue   = 3.2
        self.min_nz_blue   = -1.1
        self.max_nz_red    = 3
        self.min_nz_red    = -1
        self.g             = 9.81
        self.dt            = 0.15
        self.w             = None
        self.gamma         = None
        self.R             = 15
        # self.roll_rate    = np.radians(40)

        
        
        # For convenience: discrete actions for Blue
        # (negative bank change, zero, positive)
        self.blue_actions = [
                            0,  #Down
                            1,  #Left
                            2,  #Up
                            3   #Right
                            ]
                            
        # Similarly for Red
        self.red_actions =  [
                            0,  #Down
                            1,  #Left
                            2,  #Up
                            3   #Right
                            ]
                            
        self.n_action = 4
        
        # We keep an internal "state"
        # Let's define as:
        # state = [ xB, yB, headingB, bankB, xR, yR, headingR, bankR ]
        self.state = None
        self.done = False

    def reset(self, 
              xB=None, yB=None, zB = None, headingB=None, flpathB=None,
              xR=None, yR=None, zR = None, headingR=None, flpathR=None):
        """
        Resets the environment. 
        If None, randomly sample from distribution 
        """
        def sample_with_defaults(value, dist_type, *dist_args):
            if value is not None:
                return value
            if dist_type == 'normal':
                # dist_args = (mean, std)
                return np.random.normal(dist_args[0], dist_args[1])
            elif dist_type == 'uniform':
                # dist_args = (low, high)
                return np.random.uniform(dist_args[0], dist_args[1])
            else:
                raise ValueError("Unknown dist type.")
        
        # Positions ~ Normal(0,7/3) but clipped to [-boundary,boundary]
        std = 28
        xB = sample_with_defaults(xB, 'normal', 0.0, std)
        yB = sample_with_defaults(yB, 'normal', 0.0, std)
        zB = sample_with_defaults(zB, 'normal', 0.0, std/3)
        xR = sample_with_defaults(xR, 'normal', 0.0, std)
        yR = sample_with_defaults(yR, 'normal', 0.0, std)
        zR = sample_with_defaults(zR, 'normal', 0.0, std/3)

        # Headings, fligh path ~ uniform
        headingB = sample_with_defaults(headingB, 'uniform', -np.pi               , np.pi)
        headingR = sample_with_defaults(headingR, 'uniform', -np.pi               , np.pi)
        flpathB  = sample_with_defaults(flpathB,  'uniform', -self.max_flpath_blue, self.max_flpath_blue)
        flpathR  = sample_with_defaults(flpathR,  'uniform', -self.max_flpath_red , self.max_flpath_red)
        
        self.state = np.array([xB, yB, zB, headingB, flpathB,
                               xR, yR, zR, headingR, flpathR])
        self.done = False
        return self.state.copy()
    
    
    def step(self,blue_action_index,red_action_index):
        
        if self.done:
            return self.state.copy(), 0.0, True
        
        # Pack Input
        uB = self.blue_actions[blue_action_index]
        uR = self.red_actions[red_action_index]
        u = np.array([uB, uR])
        
        # Pack State
        # xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = self.state
        state = self.state.copy()
        
        for _ in range(1):
            x_dot = self.dynamics(state,u)
            next_state = state + x_dot * self.dt
            # Bound flight path angle
            next_state[4] = np.max([np.min([next_state[4],self.max_flpath_blue]),-self.max_flpath_blue])
            next_state[9] = np.max([np.min([next_state[4],self.max_flpath_red]), -self.max_flpath_red])
        
        self.state = next_state
        # 7) Compute reward
        reward = self._reward(next_state)
        
        goal_zone = self._in_goal_zone(next_state,1)
        # 8) Possibly check if Blue is in the goal zone => done
        if goal_zone == 1:
            self.done = True
            print("Blue won")
            
        elif goal_zone == 2:
            self.done = True
            print("Red won") 
            
        return next_state.copy(), reward, self.done
    
    def step_outer(self,state,blue_action_index,red_action_index,fastFlag = 0):
        
        # if self.done:
        #     return self.state.copy(), 0.0, True
        
        done = False

        # Pack Input
        uB = self.blue_actions[blue_action_index]
        uR = self.red_actions[red_action_index]
        u = np.array([uB, uR])
        
        for _ in range(1):
            x_dot = self.dynamics(state,u)
            next_state = state + x_dot * self.dt
            # Bound flight path angle
            next_state[4] = np.max([np.min([next_state[4],self.max_flpath_blue]),-self.max_flpath_blue])
            next_state[9] = np.max([np.min([next_state[9],self.max_flpath_red]), -self.max_flpath_red])    
        
        if fastFlag == 1:   #no computing reward or in_goal_zone
            reward = 0
            
        elif fastFlag == 2: #no computing reward
            reward = 0 
            if self._in_goal_zone(next_state):
                done = True
        else:
            # 7) Compute reward
            reward = self._reward(next_state)
            
            # 8) Possibly check if Blue is in the goal zone => done
            if self._in_goal_zone(next_state):
                done = True
        
        return next_state.copy(), reward, done
            
    
    # -----------------------------
    # Internal Helpers
    # -----------------------------
    def _reward(self, state):
        """
        Combined reward: 0.8*g_pa + 0.2*S
        - g_pa is discrete (1 if in goal zone, else 0)
        - S is continuous
        """
        g_pa_val = 1.0 if self._in_goal_zone(state) else 0.0
        s_val = self.S_function(state)
        return 0.8*g_pa_val + 0.2*s_val

    def _in_goal_zone(self, state, flag = 0):
        """
        Check if Blue is behind Red with certain geometry constraints 
        (AA, ATA in certain ranges, distance in certain range, etc.).
        """

        R   = self.R_find(state)
        AA_b  = self.AA_find(state,0)
        ATA_b = self.ATA_find(state,0)
        
        AA_r  = self.AA_find(state,1)
        ATA_r = self.ATA_find(state,1)
        
        if (R <= self.R) and (R >= 0.1) and (abs(AA_b) < 60) and (abs(ATA_b) < 30):
            return 1 #blue won
        elif (flag) and (R <= self.R) and (R >= 0.1) and (abs(AA_r) < 60) and (abs(ATA_r) < 30):
            return 2 #red won
        else:
            return 0

    def _red_minimax(self, state):
        """
        For each possible 3-step sequence of Red actions, 
        we approximate that Blue picks actions that minimize S. 
        We'll pick the Red action that yields the highest final S 
        after 3 steps, ignoring discount for that short horizon.  
        """
        
        state_org = state
        S_max = -999999999
        n_lookahead = 3
        for red_action_index in range(self.n_action):
            
            state = state_org
            S_min = 999999999
            for blue_action_index in range(self.n_action):
                S = 0
                for i in range(n_lookahead):
                    # simulate n step with this red action
                    state, _, _ = self.step_outer(state, blue_action_index, red_action_index,1)
                    S += self.S_function(state,1)
                if S < S_min:
                    S_min = S
                    #red_minimax = red_action_index
            if S_min > S_max:
                red_minimax = red_action_index
                S_max = S_min
                
        return red_minimax
    
    def _blue_minimax(self, state):
        """
        For each possible 3-step sequence of Red actions, 
        we approximate that Blue picks actions that minimize S. 
        We'll pick the Red action that yields the highest final S 
        after 3 steps, ignoring discount for that short horizon.  
        """
        
        state_org = state
        S_max = -999999999
        n_lookahead = 3
        for blue_action_index in range(self.n_action):
            
            state = state_org
            S_min = 999999999
            for red_action_index in range(self.n_action):
                S = 0
                for i in range(n_lookahead):
                    # simulate n step with this red action
                    state, _, _ = self.step_outer(state, blue_action_index, red_action_index,1)
                    S += self.S_function(state,0)
                if S < S_min:
                    S_min = S
            if S_min > S_max:
                blue_minimax = blue_action_index
                S_max = S_min
                
        return blue_minimax
    
    def blue_nom_action(self,state):
        
        best_blue_action = None
        best_val = -1e9
        for action in self.blue_actions:

            state_next, reward, done = self.simulate_step(state, action)

            val_next = 0.0 if done else np.dot(self.w, self.feature_func(state_next))

            q_val = reward + self.gamma * val_next
            if q_val > best_val:
                best_val = q_val
                best_blue_action = action
                
        return best_blue_action
                
    def simulate_step(self, state, blue_action):
        """
        A helper function that simulates one step from state s 
        if Blue picks 'blue_action' and Red picks the best 3-step lookahead (approx).
        Return: next_state, immediate_reward, done_flag
        """
        red_best_action = self._red_3step_lookahead(state, blue_action)
        
        state_next, reward, done = self.step_outer(state,blue_action,red_best_action)
            
        return state_next, reward, done
    
    def collectSample(self,nsample):
        """
        A helper function that simulates one step from state s 
        if Blue picks 'blue_action' and Red picks the best 3-step lookahead (approx).
        Return: next_state, immediate_reward, done_flag
        """
        sampleX = []
        n = nsample
        
        while len(sampleX) < n:
            
            state = self.reset()  # random
            sampleX.append(state)
            done = False
            while not done:
                
                blue_minimax_action = self._blue_minimax(state)                
                red_minimax_action  = self._red_minimax(state)
            
                state, _, done = self.step_outer(state,blue_minimax_action,red_minimax_action,2)
                sampleX.append(state)
            
        print(str(nsample)+ 'samples collected.')
        return sampleX
        
    def ATA_find(self,state,relFlag = 0):
        
        xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state
        
        if relFlag == 0:   # rel to blue
            
            LOS_vec = np.array([xR - xB , yR - yB, zR - zB]) 
            magnitude = np.linalg.norm(LOS_vec) + 0.0000001
            LOS_vec = LOS_vec/magnitude
            
            Vel_vec = np.array([np.cos(headingB)*np.cos(flpathB), np.sin(headingB)*np.cos(flpathB), np.sin(flpathB)])
            
            ATA = np.arccos(min(max(np.dot(LOS_vec,Vel_vec),-1),1))
            return np.degrees(ATA)
        
        else:              # rel to red
            LOS_vec = np.array([xB - xR , yB - yR, zB - zR]) 
            magnitude = np.linalg.norm(LOS_vec) + 0.0000001
            LOS_vec = LOS_vec/magnitude
            
            Vel_vec = np.array([np.cos(headingR)*np.cos(flpathR), np.sin(headingR)*np.cos(flpathR), np.sin(flpathR)])
            
            ATA = np.arccos(min(max(np.dot(LOS_vec,Vel_vec),-1),1))
            return np.degrees(ATA)

    def AA_find(self,state,relFlag = 0):
        
        xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state
        
        if relFlag == 0:   # rel to blue
            
            inv_LOS_vec = np.array([xB - xR , yB - yR, zB - zR]) 
            magnitude = np.linalg.norm(inv_LOS_vec) + 0.0000001
            inv_LOS_vec = inv_LOS_vec/magnitude
            
            inv_red_Vel_vec = -np.array([np.cos(headingR)*np.cos(flpathR), np.sin(headingR)*np.cos(flpathR), np.sin(flpathR)])
            
            AA = np.arccos(min(max(np.dot(inv_LOS_vec,inv_red_Vel_vec),-1),1))
            return np.degrees(AA)
        
        else:              # rel to red
            inv_LOS_vec = np.array([xR - xB , yR - yB, zR - zB])  
            magnitude = np.linalg.norm(inv_LOS_vec)  + 0.0000001
            inv_LOS_vec = inv_LOS_vec/magnitude
            
            inv_blue_Vel_vec = -np.array([np.cos(headingB)*np.cos(flpathB), np.sin(headingB)*np.cos(flpathB), np.sin(flpathB)])
            
            AA = np.arccos(min(max(np.dot(inv_LOS_vec,inv_blue_Vel_vec),-1),1))
            return np.degrees(AA)
        
    def HCA_find(self,state):
        
        xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state
        
        blue_Vel_vec = np.array([np.cos(headingB)*np.cos(flpathB), np.sin(headingB)*np.cos(flpathB), np.sin(flpathB)])
        red_Vel_vec  = np.array([np.cos(headingR)*np.cos(flpathR), np.sin(headingR)*np.cos(flpathR), np.sin(flpathR)])
        
        HCA = np.arccos(min(max(np.dot(blue_Vel_vec,red_Vel_vec),-1),1))
        return np.degrees(HCA)
    
    def R_find(self,state):
        
        xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state
        
        R = np.array([xB - xR , yB - yR, zB - zR])
        R_magnitude = np.linalg.norm(R)

        return R_magnitude
    
    def ClosureAngle_find(self,state):
        
        xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state

        
        LOS_vec = np.array([xR - xB , yR - yB, zR - zB])  
        magnitude = np.linalg.norm(LOS_vec) + 0.0000001
        LOS_vec = LOS_vec/magnitude    
        
        Vel_blue_vec = np.array([np.cos(headingB)*np.cos(flpathB), np.sin(headingB)*np.cos(flpathB), np.sin(flpathB)])
        Vel_red_vec  = np.array([np.cos(headingR)*np.cos(flpathR), np.sin(headingR)*np.cos(flpathR), np.sin(flpathR)])
        
        Vel_rel_vec = Vel_blue_vec - Vel_red_vec 
        magnitude = np.linalg.norm(Vel_rel_vec) + 0.0000001
        Vel_rel_vec = Vel_rel_vec/magnitude    

        ClosureAngle = np.arccos(min(max(np.dot(LOS_vec,Vel_rel_vec),-1),1))
        
        return np.degrees(ClosureAngle)
    
    def SA_find(self,state):
        
        AA = self.AA_find(state,0)
        ATA = self.ATA_find(state,0)
        
        SA = 1 - ( (1- AA/180) + (1 - ATA/180) )
        
        return SA
    
    def SR_find(self,state):
        k = 0.1
        Rd = self.R
        
        R = self.R_find(state)
        SR = np.exp(-(abs(R-Rd)/(180*k)))
        
        return SR

    def S_function(self,state,relFlag = 0):
        
        k = 0.1
        Rd = self.R
        
        R   = self.R_find(state)

        if relFlag == 0:
            AA  = self.AA_find(state,0)
            ATA = self.ATA_find(state,0)
        else:
            AA  = self.AA_find(state,1)
            ATA = self.ATA_find(state,1) 
            
        S = ( ( (1- AA/180) + (1-ATA/180) ) / 2  ) * (np.exp(-(abs(R-Rd)/(180*k))))
        
        return S
        
    # -----------------------------
    # 2. Feature Extraction
    # -----------------------------

    def extract_features_13(self, state, prev_state= np.array([None,None])):
        """
        We have 11 base features (no bias):
        1) |AA|
        2) Distance
        3) max(0, AA)
        4) min(0, ATA)
        5) SA
        6) SR
        7) |HCA|
        8) (10 - |AA Rate|)
        9) ATA Rate
        10) (10 - |ATA Rate|)
        11) Closure Angle
        12) Red Bank angle
        13) Blue Bank angle
        """
        # xB, yB, zB, headingB, flpathB, xR, yR, zR, headingR, flpathR = state

        if any(prev_state == None):
            prev_state = state
            
        AA           = self.AA_find(state,0)
        AA_prev      = self.AA_find(prev_state,0)
        ATA          = self.ATA_find(state,0)
        ATA_prev     = self.ATA_find(prev_state,0)
        HCA          = self.HCA_find(state)
        ClosureAngle = self.ClosureAngle_find(state)
        R            = self.R_find(state)
        SA           = self.SA_find(state)
        SR           = self.SR_find(state)
        
        AA_rate =  (AA  - AA_prev)  / ((self.dt)*1)
        ATA_rate = (ATA - ATA_prev) / ((self.dt)*1)
        
        feat1  = abs(AA)
        feat2  = R
        feat3  = max(0,AA)
        feat4  = min(0,ATA)
        feat5  = SA
        feat6  = SR
        feat7  = abs(HCA)
        feat8  = 10 - abs(AA_rate)
        feat9  = ATA_rate
        feat10 = 10 - abs(ATA_rate)
        feat11 = ClosureAngle
        # feat12 = np.degrees(bR)
        # feat13 = np.degrees(bB)
        
        features = np.array([
            feat1,  feat2,  feat3, feat4, feat5,
            feat6,  feat7,  feat8, feat9, feat10,
            feat11
                            ])

        return features

    def second_order_no_bias(self,base_feat):
        """
        Expand an 13-dim base feature vector 
        to second-order polynomial terms with NO constant term.
        That is:
        [ x1, x2, ..., x11, 
            x1^2, x2^2, ..., x11^2,
            x1*x2, x1*x3, ..., x10*x11 ]
        Total = 13 + 13 + 78 = 104.
        """
        n = len(base_feat)  # 13
        # linear
        phi_lin = base_feat
        # squares
        phi_sq = base_feat**2
        # cross terms
        cross_terms = []
        for i in range(n):
            for j in range(i+1, n):
                cross_terms.append(base_feat[i] * base_feat[j])
        phi_cross = np.array(cross_terms, dtype=base_feat.dtype)
        
        return np.concatenate([phi_lin, phi_sq, phi_cross], axis=0)

    # Wrapper to get the final 104-dim feature
    def feature_func(self,state, prev_state = np.array([None,None])):
        base = self.extract_features_13(state, prev_state=prev_state)
        return self.second_order_no_bias(base)

    def dynamics(self,x,u):
        """
        Computes the time derivatives (dx) for the blue and red aircraft motion equations 
        given the current state and control inputs.

        Returns
        -------
        dx : ndarray of length 10
            Time derivatives [Bdx/dt, Bdy/dt, Bdz/dt, Bdpsi/dt, Bdgamma/dt,
                              Rdx/dt, Rdy/dt, Rdz/dt, Rdpsi/dt, Rdgamma/dt]
        """
        
        # Unpack current states
        _, _, _, headingB, flpathB, _, _, _, headingR, flpathR = x
        
        # Find control inputs
        uB , uR = u[0:2]
        
        if (uB == 0) | (uB == 2):  # Down or Up
            bank_B = 0
            nz_B   = self.max_nz_blue * (uB - 1)
            
        else:                      # Left or Right
            bank_B = self.max_bank_blue * (uB - 2)
            nz_B   = np.cos(flpathB)/np.cos(bank_B)
            
        if (uR == 0) | (uR == 2):
            bank_R = 0
            nz_R   = self.max_nz_red * (uR -1)
            
        else:
            bank_R = self.max_bank_red * (uR - 2)
            nz_R   = np.cos(flpathR)/np.cos(bank_R)    
        
        # Constant Velocity
        V = self.velocity
        g = self.g

        # Compute derivatives
        Bdx_dt      = V * np.cos(headingB) * np.cos(flpathB)               # ẋ
        Bdy_dt      = V * np.sin(headingB) * np.cos(flpathB)               # ẏ
        Bdz_dt      = V * np.sin(flpathB)                                  # ḣ
        Bdpsi_dt    = (g * nz_B * np.sin(bank_B)) / (V * np.cos(flpathB))  # ψ̇
        Bdgamma_dt  = (g / V) * (nz_B * np.cos(bank_B) - np.cos(flpathB))  # γ̇
        Rdx_dt      = V * np.cos(headingR) * np.cos(flpathR)               # ẋ
        Rdy_dt      = V * np.sin(headingR) * np.cos(flpathR)               # ẏ
        Rdz_dt      = V * np.sin(flpathR)                                  # ḣ
        Rdpsi_dt    = (g * nz_R * np.sin(bank_R)) / (V * np.cos(flpathR))  # ψ̇
        Rdgamma_dt  = (g / V) * (nz_R * np.cos(bank_R) - np.cos(flpathR))  # γ̇

        # Pack into a single array for the ODE solver
        dx = np.array([Bdx_dt, Bdy_dt, Bdz_dt, Bdpsi_dt, Bdgamma_dt, 
                       Rdx_dt, Rdy_dt, Rdz_dt, Rdpsi_dt, Rdgamma_dt])
        
        return dx