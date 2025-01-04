import numpy as np



def dynamics(x, u, g=9.81):
    """
    Computes the time derivatives (dx) for the aircraft motion equations 
    given the current state and control inputs.

    Parameters
    ----------
    t : float
        Current time (not used in these equations but included for ODE solvers).
    state : array-like of length 6
        [x, y, h, V, psi, gamma]
        x     : inertial x-position (m)
        y     : inertial y-position (m)
        h     : altitude (m)
        V     : speed (m/s)
        psi   : heading angle (rad)
        gamma : flight path angle (rad)
    controls : array-like of length 3
        [nx, nz, mu]
        nx : longitudinal load factor
        nz : normal load factor
        mu : bank angle (rad)
    g : float, optional
        Gravitational acceleration (m/s^2), default 9.81

    Returns
    -------
    dx : ndarray of length 6
        Time derivatives [dx/dt, dy/dt, dh/dt, dV/dt, dpsi/dt, dgamma/dt]
    """
    
    # Unpack current states
    x, y, h, V, psi, gamma = x

    # Unpack controls
    nx, nz, mu = u

    # Compute derivatives
    dx_dt      = V * np.cos(psi) * np.cos(gamma)              # ẋ
    dy_dt      = V * np.sin(psi) * np.cos(gamma)              # ẏ
    dh_dt      = V * np.sin(gamma)                            # ḣ
    dV_dt      = g * (nx - np.sin(gamma))                     # V̇
    dpsi_dt    = (g * nz * np.sin(mu)) / (V * np.cos(gamma))  # ψ̇
    dgamma_dt  = (g / V) * (nz * np.cos(mu) - np.cos(gamma))  # γ̇

    # Pack into a single array for the ODE solver
    dx = np.array([dx_dt, dy_dt, dh_dt, dV_dt, dpsi_dt, dgamma_dt])
    
    return dx

print(dynamics([1,2,3,4,5,6],[1,2,3]))