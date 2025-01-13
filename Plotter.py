import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

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
        # # Mark the first point with a star
        ax.scatter(red_x[0], red_y[0], red_z[0], color='red', marker='*', s=100, label='Red Start')
        
        # Labeling
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("Blue vs. Red Position History (3D)")
        ax.legend()
        plt.show()
    
    else:
        # -- 2D plotting (original code) --
        # Extract x, y for Blue (B) and Red (R)
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
        plt.axis('equal')
        plt.show()
