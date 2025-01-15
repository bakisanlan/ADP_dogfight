import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np

def set_axes_equal(ax):
    """
    Make the 3D axes of a 3D plot have equal scale so that spheres appear as spheres.
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

def draw_fov_cone_3d(ax, x, y, z, heading_deg, flightpath_deg, 
                     fov_deg=30.0, length=3.0, num_circle_pts=20, color = 'blue'):
    """
    Draw a wireframe "FOV cone" in 3D at position (x,y,z), 
    oriented by heading_deg (yaw) & flightpath_deg (pitch).
    
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        Matplotlib 3D axes.
    x, y, z : float
        Aircraft's position.
    heading_deg : float
        Heading (yaw) angle in degrees (0 deg usually along +X, increasing CCW).
    flightpath_deg : float
        Flight path (pitch) angle in degrees (0 deg = level flight, + up).
    fov_deg : float
        Full cone angle for the FOV (we split it in half on each side).
    length : float
        How far the cone extends from (x,y,z).
    num_circle_pts : int
        Number of points to generate the cone perimeter.
    """
    # Convert angles to radians

    h = heading_deg      # heading (yaw)
    fp = flightpath_deg - np.pi/2 # flight path (pitch)
    half_angle = fov_deg / 2.0

    # Rotation about Z by heading: Rz(heading)
    ch, sh = np.cos(h), np.sin(h)
    Rz = np.array([[ ch, -sh,  0],
                   [ sh,  ch,  0],
                   [  0,   0,  1]])
    
    # Rotation about Y by (-flightpath) to tilt up/down:
    cfp, sfp = np.cos(-fp), np.sin(-fp)
    Ry = np.array([[ cfp, 0, sfp],
                   [   0, 1,   0],
                   [-sfp, 0, cfp]])
    
    # Combine the rotations (apply Ry then Rz in that order)
    R = Rz @ Ry  # Overall rotation matrix
    
    # Generate a ring of points around the cone's surface in LOCAL coordinates.
    # In local coords, the "forward" direction is +Z, so we create points that are
    # half_angle away from +Z. We'll param by angle phi around that ring.
    phi_vals = np.linspace(0, 2*np.pi, num_circle_pts, endpoint=True)
    
    # Container for the ring (in global coords)
    ring_points = []
    
    # The local 'tip' of the cone is at the origin [0,0,0], 
    # and the cone extends in +Z.  We'll draw a ring at Z=1 with radius = tan(half_angle).
    # Then later, we scale by "length" in the Z-direction.
    for phi in phi_vals:
        # local_x = radius * cos(phi)
        # local_y = radius * sin(phi)
        # local_z = 1.0
        # where radius = tan(half_angle)
        r = np.tan(half_angle)
        lx = r * np.cos(phi)
        ly = r * np.sin(phi)
        lz = 1.0
        
        # Convert to 3D vector
        local_vec = np.array([lx, ly, lz])
        
        # Rotate to global, then scale by `length`, then shift by (x,y,z)
        global_vec = R @ local_vec
        # The tip is at aircraft position, ring is at tip + length*(global_vec).
        ring_pt = np.array([x, y, z]) + length * global_vec
        ring_points.append(ring_pt)
    
    ring_points = np.array(ring_points)
    
    # Draw wireframe lines:
    #  1) perimeter of the ring
    #  2) lines from tip (x,y,z) to each point on the ring
    for i in range(len(ring_points)):
        i_next = (i + 1) % len(ring_points)
        # perimeter
        ax.plot(
            [ring_points[i, 0], ring_points[i_next, 0]],
            [ring_points[i, 1], ring_points[i_next, 1]],
            [ring_points[i, 2], ring_points[i_next, 2]],
            color= color,
            alpha = 0.3
        )
        # radial line from tip
        ax.plot(
            [x, ring_points[i, 0]],
            [y, ring_points[i, 1]],
            [z, ring_points[i, 2]],
            color= color,
            alpha = 0.3
        )

def plot_trajectories(states, flag3D=False):
    """
    Given a list of states over time:
       - In 2D mode (flag3D=False), we assume each state is at least:
            [xB, yB, headingB, bankB, xR, yR, headingR, bankR]
         and we only use xB, yB for Blue and xR, yR for Red.
         
       - In 3D mode (flag3D=True), we assume each state is at least:
            [xB, yB, zB, headingB, flightPathB, xR, yR, zR, headingR, flightPathR]
         so that we can plot xB, yB, zB for Blue and xR, yR, zR for Red.
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
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Blue trajectory
        ax.plot(blue_x, blue_y, blue_z, 'o', color='blue', label='Blue Trajectory')
        ax.scatter(blue_x[0], blue_y[0], blue_z[0], color='blue', marker='*', s=100, label='Blue Start')
        
        # Plot Red trajectory
        ax.plot(red_x, red_y, red_z, 'o', color='red', label='Red Trajectory')
        ax.scatter(red_x[0], red_y[0], red_z[0], color='red', marker='*', s=100, label='Red Start')
        
        # ---------------------------------------------------------------------
        # ADD THE FOV CONES FOR BOTH AIRCRAFT (FINAL STATES ONLY)
        # ---------------------------------------------------------------------
        # Get the final Blue aircraft data:
        xB_fin   = states[-1][0]
        yB_fin   = states[-1][1]
        zB_fin   = states[-1][2]
        hdgB_fin = states[-1][3]   # heading
        fpB_fin  = states[-1][4]   # flight path angle
        
        # Get the final Red aircraft data:
        xR_fin   = states[-1][5]
        yR_fin   = states[-1][6]
        zR_fin   = states[-1][7]
        hdgR_fin = states[-1][8]   # heading
        fpR_fin  = states[-1][9]   # flight path angle
        
        # Draw 30-deg FOV cones of length ~200 (adjust as needed)
        draw_fov_cone_3d(ax, xB_fin, yB_fin, zB_fin, hdgB_fin, fpB_fin, 
                         fov_deg=30.0, length= 1.2, num_circle_pts=100 , color = 'blue')
        draw_fov_cone_3d(ax, xR_fin, yR_fin, zR_fin, hdgR_fin, fpR_fin, 
                         fov_deg=30.0, length=1.2, num_circle_pts=100,color= 'red')
        # ---------------------------------------------------------------------

        # Labeling
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("Blue vs. Red Position History (3D)")
        ax.legend()
        
        # Enforce equal aspect ratio in 3D
        set_axes_equal(ax)
        
        # Often for flight data, you might want to invert Y or Z, 
        # but that depends on your convention:
        ax.invert_zaxis()  
        ax.invert_yaxis()

        plt.show()
    
    else:
        # -- 2D plotting --
        # We only use xB, yB, xR, yR in 2D
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
        plt.title("Blue vs. Red Position History (2D)")
        plt.grid(True)
        plt.legend()
        
        # Force equal scaling in 2D
        plt.axis('equal')
        
        plt.show()
