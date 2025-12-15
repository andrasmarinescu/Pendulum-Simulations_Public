import numpy as np
import matplotlib.pyplot as plt

def plot_phase_space(sol, labels=('x', 'y'), title=None, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(sol[:,0], sol[:,1], lw=0.8)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if title:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

def energy_simple_pendulum(sol):
    # sol columns: theta, omega
    theta = sol[:,0]
    omega = sol[:,1]
    KE = 0.5 * omega**2
    PE = 1 - np.cos(theta)
    return KE + PE