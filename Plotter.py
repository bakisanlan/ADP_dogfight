import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np

def set_axes_equal(ax):
    """
    Make the 3D axes of a 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.  
    This is one solution to Matplotlib's 'equal axis' problem for 3D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    
    max_range = max(x_range, y_range, z_range)
    
    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])
    
    ax.set_xlim3d([x_mid - 0.5 * max_range, x_mid + 0.5 * max_range])
    ax.set_ylim3d([y_mid - 0.5 * max_range, y_mid + 0.5 * max_range])
    ax.set_zlim3d([z_mid - 0.5 * max_range, z_mid + 0.5 * max_range])

def plot_trajectories(states, flag3D=False):
    """
    Given a list of states over time:
       - In 2D mode (flag3D=False), we assume each state is at least:
            [xB, yB, headingB, bankB, xR, yR, headingR, bankR]
         and we only use xB, yB for Blue and xR, yR for Red.
         
       - In 3D mode (flag3D=True), we assume each state is at least:
            [xB, yB, zB, headingB, bankB, xR, yR, zR, headingR, bankR]
         so that we can plot xB, yB, zB for Blue and xR, yR, zR for Red.
    
    This function plots the trajectories in 2D or 3D depending on the flag.
    """
    
    if flag3D:
        # -- 3D plotting --
        # Extract x, y, z for Blue (B) and Red (R)
        blue_x = [s[0] for s in states]
        blue_y = [s[1] for s in states]
        blue_z = [s[2] for s in states]
        
        red_x  = [s[5] for s in states]
        red_y  = [s[6] for s in states]
        red_z  = [s[7] for s in states]
        
        # Create 3D figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Blue trajectory
        ax.plot(blue_x, blue_y, blue_z, 'o', color='blue', label='Blue Trajectory')
        # Mark the first point with a star
        ax.scatter(blue_x[0], blue_y[0], blue_z[0], color='blue', marker='*', s=100, label='Blue Start')
        
        # Plot Red trajectory
        ax.plot(red_x, red_y, red_z, 'o', color='red', label='Red Trajectory')
        ax.scatter(red_x[0], red_y[0], red_z[0], color='red', marker='*', s=100, label='Red Start')
        
        # Labeling
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("Blue vs. Red Position History (3D)")
        ax.legend()
        
        # *** IMPORTANT: Enforce equal aspect ratio ***
        set_axes_equal(ax)
        ax.invert_zaxis()
        ax.invert_yaxis()


        plt.show()
    
    else:
        # -- 2D plotting --
        blue_x = [s[0] for s in states]  # xB
        blue_y = [s[1] for s in states]  # yB
        red_x  = [s[4] for s in states]  # xR
        red_y  = [s[5] for s in states]  # yR
        
        plt.figure(figsize=(6, 6))
        
        # Plot Blue trajectory
        plt.plot(blue_x, blue_y, '-o', color='blue', label='Blue Trajectory')
        plt.plot(blue_x[0], blue_y[0], 'b*', markersize=10, label='Blue Start')
        
        # Plot Red trajectory
        plt.plot(red_x, red_y, '-o', color='red', label='Red Trajectory')
        plt.plot(red_x[0], red_y[0], 'r*', markersize=10, label='Red Start')
        
        # Labeling and grid
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Blue vs. Red Position History")
        plt.grid(True)
        plt.legend()
        
        # Force equal scaling in 2D
        plt.axis('equal')
        
        plt.show()
