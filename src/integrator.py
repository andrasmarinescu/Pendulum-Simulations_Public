
import numpy as np

def integrate_leapfrog(func, y0, t, dt, args=()):
    """
    General symplectic leapfrog integrator (odeint-compatible API).

    Parameters
    ----------
    func : callable
        Function f(y, t, *args) returning dy/dt as 1D array of len(y0).
        First half of y are "positions" q, second half "momenta" p.
    y0 : array_like
        Initial state vector [q1, ..., qN, p1, ..., pN].
        Must have even length.
    t : array_like
        Times at which to output the solution.
    args : tuple, optional
        Additional parameters for func.
    dt : float, optional
        Internal integration timestep. Defaults to min(diff(t)) / 10.

    Returns
    -------
    y_out : ndarray
        Array of shape (len(t), len(y0)) containing y(t) at each output time.
    """

    t = np.asarray(t, dtype=float)
    t.sort()
    t_end = t[-1]
    n_out = len(t)

    y0 = np.asarray(y0, dtype=float)
    ndim = len(y0)
    if ndim % 2 != 0:
        raise ValueError("y0 must have even length: [q1, ..., qN, p1, ..., pN]")

    n_dim = ndim // 2
    y_out = np.zeros((n_out, ndim))

    # Split into position and momentum arrays
    q = np.array(y0[:n_dim], dtype=float)
    p = np.array(y0[n_dim:], dtype=float)

    # Initial half-step momentum update
    dydt = np.asarray(func(np.concatenate([q, p]), 0.0, *args), dtype=float)
    dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
    p += 0.5 * dt * dpdt

    t_curr = 0.0
    next_index = 0

    # Integration loop
    for step in range(int(t_end / dt) + 2):
        # Output if reached/passed next requested time
        while next_index < n_out and t_curr >= t[next_index]:
            dydt = np.asarray(func(np.concatenate([q, p]), t_curr, *args), dtype=float)
            dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
            y_out[next_index, :n_dim] = q
            y_out[next_index, n_dim:] = p - 0.5 * dt * dpdt  # back to integer step
            next_index += 1
            if next_index == n_out:
                break
        if next_index == n_out:
            break

        # Full leapfrog update
        q += dt * p
        dydt = np.asarray(func(np.concatenate([q, p]), t_curr + 0.5 * dt, *args), dtype=float)
        dqdt, dpdt = dydt[:n_dim], dydt[n_dim:]
        p += dt * dpdt
        t_curr += dt

    return y_out

from scipy.integrate import solve_ivp

def integrate_generic(rhs, y0, t, dt=None, args=()):
    from scipy.integrate import solve_ivp
    import numpy as np

    sol = solve_ivp(
        fun=lambda tt, yy: rhs(yy, tt, *args),
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        rtol=1e-9,
        atol=1e-9,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y.T



def wrap_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi