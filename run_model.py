import argparse
import numpy as np
from src.integrator import integrate_leapfrog, integrate_generic
import src.models as M
import src.utils as U
import matplotlib.pyplot as plt

MODEL_MAP = {
    'simple': (
        M.simple_pendulum,
        integrate_generic,
        np.array([2.0, 0.0]),     # y0
        ()                        # params
    ),
    'damped': (
        M.damped_pendulum,
        integrate_generic,
        np.array([1.0, 0.0]),
        ()
    ),
    'driven': (
        M.driven_pendulum,
        integrate_generic,
        np.array([0.2, 0.0]),
        ()
    ),
    'double': (
        M.double_pendulum,
        integrate_generic,
        np.array([4.0, 0.0, 2.0, 1.0]),
        ()
    ),
    'cart': (
        M.driven_cart_pendulum,
        integrate_leapfrog,
        np.array([0.10, 1.8, 0.10, 0.0]), 
        (0.4, 10.0)                      
    )
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='simple')
    parser.add_argument('--tmax', type=float, default=100.0)
    parser.add_argument('--dt', type=float, default=0.0001)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()


    rhs, integrator, y0, params = MODEL_MAP[args.model]
    t = np.arange(0, args.tmax, args.dt)

    print(f"Running {args.model} with integrator {integrator.__name__}")
    # sol = integrator(rhs, np.array(y0), t, args.dt)
    sol = integrator(
        rhs,
        y0,
        t,
        args.dt,
        args=params
    )

    # Plotting: for single-DOF models plot phase space, for double pendulum plot angle vs time
    if args.model == 'double':
        theta1 = sol[:,0]
        omega1 = sol[:,1]
        theta2 = sol[:,2]
        omega2 = sol[:,3]

        # Time series plot
        plt.figure(figsize=(8,4))
        plt.plot(t, theta1, label='theta1')
        plt.plot(t, theta2, label='theta2')
        plt.xlabel('t')
        plt.ylabel('theta (rad)')
        plt.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save + "_time.png", dpi=200)
        else:
            plt.show()

        # Poincare section
        sec_th2 = []
        sec_om2 = []

        for i in range(len(theta1) - 1):
            if theta1[i] < 0 and theta1[i+1] > 0 and omega1[i] > 0:
                frac = -theta1[i] / (theta1[i+1] - theta1[i])

                th2_cross = theta2[i] + frac * (theta2[i+1] - theta2[i])
                om2_cross = omega2[i] + frac * (omega2[i+1] - omega2[i])

                th2_cross = (th2_cross + np.pi) % (2*np.pi) - np.pi

                sec_th2.append(th2_cross)
                sec_om2.append(om2_cross)

        plt.figure(figsize=(5,5))
        plt.scatter(sec_th2, sec_om2, s=6)
        plt.xlabel(r'$\theta_2$')
        plt.ylabel(r'$\omega_2$')
        plt.title(r'Poincaré Section: $\theta_1=0,\ \dot{\theta}_1>0$')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if args.save:
            plt.savefig(args.save + "_poincare.png", dpi=200)
        else:
            plt.show()
    if args.model == 'cart':
        X = sol[:,0]
        theta = sol[:,1]
        pX = sol[:,2]
        pTheta = sol[:,3]
        mu, Gamma = params

        # Reconstruct velocities
        cos = np.cos(theta)
        Delta = 1.0 - mu**2 * cos**2
        X_dot = (pX - mu*cos*pTheta) / Delta

        sec_th = []
        sec_pTh = []

        for i in range(len(X) - 1):
            # Poincaré section: X = 0, X_dot > 0
            if X[i] < 0 and X[i+1] > 0 and X_dot[i] > 0:
                frac = -X[i] / (X[i+1] - X[i])

                theta_cross = theta[i] + frac * (theta[i+1] - theta[i])
                theta_cross = (theta_cross + np.pi) % (2*np.pi) - np.pi

                pTheta_cross = (
                    pTheta[i] + frac * (pTheta[i+1] - pTheta[i])
                )

                sec_th.append(theta_cross)
                sec_pTh.append(pTheta_cross)

        plt.figure(figsize=(5,5))
        plt.scatter(sec_th, sec_pTh, s=5)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$p_\theta$')
        plt.title(r'Poincaré Section: $X=0,\ \dot X>0$')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if args.save:
            plt.savefig(args.save, dpi=200)
        else:
            plt.show()
    else:
        U.plot_phase_space(sol, labels=('theta','omega'), title=f'{args.model} phase space', save_path=args.save)
    print("Done.")

if __name__ == "__main__":
    main()