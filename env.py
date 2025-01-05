import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# -----------------------------
# 1. Environment Definition
# -----------------------------
class AirCombatEnv:
    """
    A simplified 2D air combat environment with:
      - 2 aircraft (Blue, Red)
      - State includes positions (x,y), headings, bank angles, etc.
      - Blue controls: {LEFT, STRAIGHT, RIGHT} [discrete]
      - Red does a 3-step lookahead trying to maximize S, 
        while Blue tries to minimize S.
    """
    def __init__(self, 
                 boundary=60.0,
                 velocity=2.5,
                 max_bank_blue =np.radians(23),
                 max_bank_red  =np.radians(18)
                 ):
        self.boundary = boundary
        self.velocity = velocity
        self.max_bank_blue = max_bank_blue
        self.max_bank_red = max_bank_red
        self.g            = 9.81
        self.roll_rate    = np.radians(40)
        self.dt           = 0.05
        
        #self.gamma = gamma
        
        # For convenience: discrete actions for Blue
        # (negative bank change, zero, positive)
        self.blue_actions = [
                            0,    # LEFT
                            1,    # STRAIGHT
                            2     # RIGHT
                            ]
        # Similarly for Red
        self.red_actions =  [
                            0,    # LEFT
                            1,    # STRAIGHT
                            2     # RIGHT
                            ]
        
        # We keep an internal "state"
        # Let's define as:
        # state = [ xB, yB, headingB, bankB, xR, yR, headingR, bankR ]
        self.state = None
        self.done = False

    def reset(self, 
              xB=None, yB=None, headingB=None, bankB=None,
              xR=None, yR=None, headingR=None, bankR=None):
        """
        Resets the environment. 
        If None, randomly sample from distribution 
        specified in your instructions.
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
        
        # Positions ~ Normal(0,7) but clipped to [-boundary,boundary]
        xB = sample_with_defaults(xB, 'normal', 0.0, 7.0)
        yB = sample_with_defaults(yB, 'normal', 0.0, 7.0)
        xR = sample_with_defaults(xR, 'normal', 0.0, 7.0)
        yR = sample_with_defaults(yR, 'normal', 0.0, 7.0)
        
        # Headings, bank ~ uniform
        headingB = sample_with_defaults(headingB, 'uniform', -np.pi             , np.pi)
        bankB    = sample_with_defaults(bankB,    'uniform', -self.max_bank_blue, self.max_bank_blue)
        headingR = sample_with_defaults(headingR, 'uniform', -np.pi             , np.pi)
        bankR    = sample_with_defaults(bankR,    'uniform', -self.max_bank_red , self.max_bank_red)
        
        # # Clip positions to stay within boundaries (or re-sample if you prefer)
        # xB = np.clip(xB, -self.boundary, self.boundary)
        # yB = np.clip(yB, -self.boundary, self.boundary)
        # xR = np.clip(xR, -self.boundary, self.boundary)
        # yR = np.clip(yR, -self.boundary, self.boundary)
        
        self.state = np.array([xB, yB, headingB, bankB,
                               xR, yR, headingR, bankR], dtype=np.float32)
        self.done = False
        return self.state.copy()
    
    
    def step(self,blue_action_index,red_action_index):
        
        if self.done:
            return self.state.copy(), 0.0, True
        
        
        u_B = blue_action_index - 1
        u_R = red_action_index  - 1
        
        xB, yB, hB, bB, xR, yR, hR, bR = self.state

        for i in range(5):
            
            bB = np.max([np.min([bB + u_B*self.roll_rate*self.dt ,self.max_bank_blue]), -self.max_bank_blue])
            bR = np.max([np.min([bR + u_R*self.roll_rate*self.dt ,self.max_bank_red ]), -self.max_bank_red ])

            yaw_rate_B = (self.g/self.velocity)*np.tan(bB)
            yaw_rate_R = (self.g/self.velocity)*np.tan(bR)
            
            hB = hB + yaw_rate_B*self.dt
            hR = hR + yaw_rate_R*self.dt
            
            xB = xB + self.velocity*np.cos(hB)*self.dt
            xR = xR + self.velocity*np.cos(hR)*self.dt
            
            yB = yB + self.velocity*np.sin(hB)*self.dt
            yR = yR + self.velocity*np.sin(hR)*self.dt           

        # 5) Check boundaries
        #    If out of boundary, we treat it as "done" and reset or 
        #    forcibly keep them in?
        if not (-self.boundary <= xB <= self.boundary and 
                -self.boundary <= yB <= self.boundary and
                -self.boundary <= xR <= self.boundary and
                -self.boundary <= yR <= self.boundary):
            # Out of boundary => done
            self.done = True
        
        # 6) Construct next_state
        next_state = np.array([xB, yB, hB, bB,
                               xR, yR, hR, bR], dtype=np.float32)
        
        self.state = next_state
        # 7) Compute reward
        reward = self._reward(next_state)
        
        # 8) Possibly check if Blue is in the goal zone => done
        if self._in_goal_zone(next_state):
            self.done = True
        
        return next_state.copy(), reward, self.done
    
    def step_outer(self,state,blue_action_index,red_action_index):
        
        # if self.done:
        #     return self.state.copy(), 0.0, True
        
        done = False
        
        u_B = blue_action_index - 1
        u_R = red_action_index  - 1
        
        xB, yB, hB, bB, xR, yR, hR, bR = state

        for i in range(5):
            
            bB = np.max([np.min([bB + u_B*self.roll_rate*self.dt ,self.max_bank_blue]), -self.max_bank_blue])
            bR = np.max([np.min([bR + u_R*self.roll_rate*self.dt ,self.max_bank_red ]), -self.max_bank_red ])

            yaw_rate_B = (self.g/self.velocity)*np.tan(bB)
            yaw_rate_R = (self.g/self.velocity)*np.tan(bR)
            
            hB = hB + yaw_rate_B*self.dt
            hR = hR + yaw_rate_R*self.dt
            
            xB = xB + self.velocity*np.cos(hB)*self.dt
            xR = xR + self.velocity*np.cos(hR)*self.dt
            
            yB = yB + self.velocity*np.sin(hB)*self.dt
            yR = yR + self.velocity*np.sin(hR)*self.dt           

        # 5) Check boundaries
        #    If out of boundary, we treat it as "done" and reset or 
        #    forcibly keep them in?
        if not (-self.boundary <= xB <= self.boundary and 
                -self.boundary <= yB <= self.boundary and
                -self.boundary <= xR <= self.boundary and
                -self.boundary <= yR <= self.boundary):
            # Out of boundary => done
            done = True
        
        # 6) Construct next_state
        next_state = np.array([xB, yB, hB, bB,
                               xR, yR, hR, bR], dtype=np.float32)
        
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

    def _in_goal_zone(self, state):
        """
        Check if Blue is behind Red with certain geometry constraints 
        (AA, ATA in certain ranges, distance in certain range, etc.).
        For demonstration, let's do a naive check—replace with real geometry if needed.
        """
        # Just pretend "goal zone" means distance < some threshold & some angle constraints
        #xB, yB, hB, bB, xR, yR, hR, bR = state
 
        R   = self.R_find(state)
        AA  = self.AA_find(state,0)
        ATA = self.ATA_find(state,0)
        
        if (R <= 3) and (R >= 0.1) and (abs(AA) < 60) and (abs(ATA) < 30):
            return True
        return False

    def _red_3step_lookahead(self, state, blue_action_index):
        """
        For each possible 3-step sequence of Red actions, 
        we approximate that Blue picks actions that minimize S. 
        We'll pick the Red action that yields the highest final S 
        after 3 steps, ignoring discount for that short horizon.  
        
        But we only need the *first* Red action from the best sequence.
        """
        
        state_org = state
        red_best_action = 0
        best_S = -999999.0
        n_lookahead = 3
        for red_action_index in self.red_actions:
            
            state = state_org
            S = 0
            
            for i in range(n_lookahead):
                # simulate 1 step with this red action
                state, _, _ = self.step_outer(state, blue_action_index, red_action_index)
                S += self.S_function(state,1)
            if S > best_S:
                best_S = S
                red_best_action = red_action_index
                
        return red_best_action
    
        
    def ATA_find(self,state,relFlag = 0):
        
        xB, yB, hB, bB, xR, yR, hR, bR = state
        
        if relFlag == 0:   # rel to blue
            
            LOS_vec = np.array([xR - xB , yR - yB]) 
            magnitude = np.linalg.norm(LOS_vec) + 0.0000001
            LOS_vec = LOS_vec/magnitude
            
            Vel_vec = np.array([np.cos(hB), np.sin(hB)])
            
            ATA = np.arccos(min(max(np.dot(LOS_vec,Vel_vec),-1),1))
            return np.degrees(ATA)
        
        else:              # rel to red
            LOS_vec = np.array([xB - xR , yB - yR]) 
            magnitude = np.linalg.norm(LOS_vec) + 0.0000001
            LOS_vec = LOS_vec/magnitude
            
            Vel_vec = np.array([np.cos(hR), np.sin(hR)])
            
            ATA = np.arccos(min(max(np.dot(LOS_vec,Vel_vec),-1),1))
            return np.degrees(ATA)

    def AA_find(self,state,relFlag = 0):
        
        xB, yB, hB, bB, xR, yR, hR, bR = state
        
        if relFlag == 0:   # rel to blue
            
            inv_LOS_vec = np.array([xB - xR , yB - yR]) 
            magnitude = np.linalg.norm(inv_LOS_vec) + 0.0000001
            inv_LOS_vec = inv_LOS_vec/magnitude
            
            inv_red_Vel_vec = -np.array([np.cos(hR), np.sin(hR)])
            
            AA = np.arccos(min(max(np.dot(inv_LOS_vec,inv_red_Vel_vec),-1),1))
            return np.degrees(AA)
        
        else:              # rel to red
            inv_LOS_vec = np.array([xR - xB , yR - yB]) 
            magnitude = np.linalg.norm(inv_LOS_vec)  + 0.0000001
            inv_LOS_vec = inv_LOS_vec/magnitude
            
            inv_blue_Vel_vec = -np.array([np.cos(hB), np.sin(hB)]) 
            
            AA = np.arccos(min(max(np.dot(inv_LOS_vec,inv_blue_Vel_vec),-1),1))
            return np.degrees(AA)
        
    def HCA_find(self,state):
        
        xB, yB, hB, bB, xR, yR, hR, bR = state
        
        blue_Vel_vec = np.array([np.cos(hB), np.sin(hB)]) 
        red_Vel_vec  = np.array([np.cos(hR), np.sin(hR)]) 
        
        HCA = np.arccos(min(max(np.dot(blue_Vel_vec,red_Vel_vec),-1),1))
        return np.degrees(HCA)
    
    def R_find(self,state):
        
        xB, yB, hB, bB, xR, yR, hR, bR = state
        
        R = np.array([xB - xR , yB - yR]) 
        R_magnitude = np.linalg.norm(R)

        return R_magnitude
    
    def ClosureAngle_find(self,state):
        
        xB, yB, hB, bB, xR, yR, hR, bR = state

        
        LOS_vec = np.array([xR - xB , yR - yB]) 
        magnitude = np.linalg.norm(LOS_vec) + 0.0000001
        LOS_vec = LOS_vec/magnitude    
        
        Vel_blue_vec = np.array([np.cos(hB), np.sin(hB)])
        Vel_red_vec  = np.array([np.cos(hR), np.sin(hR)])
        
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
        Rd = 2
        
        R = self.R_find(state)
        SR = np.exp(-(abs(R-Rd)/(180*k)))
        
        return SR

    def S_function(self,state,relFlag = 0):
        
        k = 0.1
        Rd = 2
        
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
        xB, yB, hB, bB, xR, yR, hR, bR = state

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
        
        AA_rate =  (AA  - AA_prev)  / ((self.dt)*5)
        ATA_rate = (ATA - ATA_prev) / ((self.dt)*5)
        
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
        feat12 = np.degrees(bR)
        feat13 = np.degrees(bB)
        
        features = np.array([
            feat1,  feat2,  feat3, feat4, feat5,
            feat6,  feat7,  feat8, feat9, feat10,
            feat11, feat12, feat13
        ], dtype=np.float32)

        return features

    def second_order_no_bias(self,base_feat):
        """
        Expand an 11-dim base feature vector 
        to second-order polynomial terms with NO constant term.
        That is:
        [ x1, x2, ..., x11, 
            x1^2, x2^2, ..., x11^2,
            x1*x2, x1*x3, ..., x10*x11 ]
        Total = 11 + 11 + 55 = 77.
        """
        n = len(base_feat)  # 11
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

    # Wrapper to get the final 77-dim feature
    def feature_func(self,state, prev_state = np.array([None,None])):
        base = self.extract_features_13(state, prev_state=prev_state)
        return self.second_order_no_bias(base)

