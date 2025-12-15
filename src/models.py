"""
Definitions of ODE right-hand-sides for each model.
Each function has f(y, t, *params) and returns dy/dt (1D array).
"""

import numpy as np

# nonlinear pendulum: y = [theta, omega]
def simple_pendulum(y, t, *params):
    theta, omega = y
    dtheta = omega
    domega = -np.sin(theta)
    return np.array([dtheta, domega])

# Damped pendulum: ddot theta = -sin(theta) - gamma * dot theta
def damped_pendulum(y, t, gamma=0.1):
    theta, omega = y
    return np.array([omega, -np.sin(theta) - gamma*omega])

# Driven pendulum: ddot theta = -sin(theta) + A cos(omega_drive t)
def driven_pendulum(y, t, A=1.2, omega_drive=2/3):
    theta, omega = y
    forcing = A * np.cos(omega_drive * t)
    return np.array([omega, -np.sin(theta) + forcing])

# Double pendulum (uses state [theta1, omega1, theta2, omega2])
def double_pendulum(y, t, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    theta1, p1, theta2, p2 = y
    
    theta1, omega1, theta2, omega2 = theta1, p1, theta2, p2

    dtheta1 = omega1
    dtheta2 = omega2

    delta = theta2 - theta1
    eps = 1e-8

    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    denom1 = np.sign(denom1) * max(abs(denom1), eps)

    denom2 = (L2 / L1) * denom1
    denom2 = np.sign(denom2) * max(abs(denom2), eps)


    domega1 = (m2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta)
               + m2 * g * np.sin(theta2) * np.cos(delta)
               + m2 * L2 * omega2 * omega2 * np.sin(delta)
               - (m1 + m2) * g * np.sin(theta1)) / denom1

    domega2 = (- m2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta)
               + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
               - (m1 + m2) * L1 * omega1 * omega1 * np.sin(delta)
               - (m1 + m2) * g * np.sin(theta2)) / denom2

    return np.array([dtheta1, domega1, dtheta2, domega2])

# cart attached to spring with pendulum hanging
def driven_cart_pendulum(y, tau, mu, Gamma):
    """
    y = [X, theta, pX, pTheta]
    """
    X, th, pX, pTh = y
    cos, sin = np.cos(th), np.sin(th)

    Delta = 1.0 - mu**2 * cos**2

    # q' = dH/dp
    Xp  = (pX  - mu*cos*pTh) / Delta
    thp = (pTh - mu*cos*pX ) / Delta

    # p' = -dH/dq
    pX_dot = -X

    pTh_dot = (
        - mu * sin / Delta**2
        * ( mu*cos*(pX**2 + pTh**2)
            - (1 + mu**2*cos**2)*pX*pTh )
        - Gamma * sin
    )

    return np.array([Xp, thp, pX_dot, pTh_dot])

  