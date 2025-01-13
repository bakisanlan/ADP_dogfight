import matplotlib.pyplot as plt


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
