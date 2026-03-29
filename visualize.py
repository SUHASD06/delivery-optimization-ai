import matplotlib.pyplot as plt
import time

def animate(env, path_history):
    plt.figure()

    for i in range(len(path_history)):
        plt.clf()

        # deliveries
        dx = [d[0] for d in env.pending_deliveries]
        dy = [d[1] for d in env.pending_deliveries]
        plt.scatter(dx, dy, c='blue', label='Deliveries')

        # fuel stations
        fx = [f[0] for f in env.fuel_stations]
        fy = [f[1] for f in env.fuel_stations]
        plt.scatter(fx, fy, c='green', label='Fuel Stations')

        # path so far
        px = [p[0] for p in path_history[:i+1]]
        py = [p[1] for p in path_history[:i+1]]
        plt.plot(px, py, c='red', marker='o', label='Path')

        # current position
        cx, cy = path_history[i]
        plt.scatter(cx, cy, c='black', s=100, label='Agent')

        plt.text(cx, cy+0.5, f"Fuel: {round(env.fuel,1)}", color='black')
        plt.text(cx, cy-0.5, f"Step {i}", color='red')

        plt.title(f"Step {i} - Delivery Simulation")
        plt.legend()
        plt.grid()

        plt.pause(0.5)

    plt.show()
