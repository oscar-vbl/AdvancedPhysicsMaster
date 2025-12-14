import numpy as np

def V(x, mw=1.0):
    '''Potential function for the path integral Monte Carlo simulation'''
    return 0.5 * mw * x**2

def euclideanAction(x_prev, x_i, x_next, dt, m=1.0, mw=1.0):
    kinetic = (m / (2 * dt)) * (
        (x_i - x_prev)**2 +
        (x_next - x_i)**2
    )
    potential = dt * V(x_i, mw)
    return kinetic + potential

def monteCarloMethod(N, n_steps, beta, dt, limit=1.0, mw=1.0, m=1.0):
    '''Perform path integral Monte Carlo simulation'''
    x = np.zeros(N)
    path = []
    x[0] = np.random.uniform(-limit, limit)
    for step in range(n_steps):
        bead = np.random.randint(0, N)
        delta_x = np.random.uniform(-limit, limit)
        x_old = x[bead]
        x_new = x_old + delta_x

        if bead > 0:     bead_prev = bead - 1
        else:            bead_prev = N - 1
        if bead < N - 1: bead_next = bead + 1
        else:            bead_next = 0
        action_old = euclideanAction(x[bead_prev], x_old, x[bead_next], dt, mw=mw, m=m)
        action_new = euclideanAction(x[bead_prev], x_new, x[bead_next], dt, mw=mw, m=m)

        r_random = np.random.rand()
        delta_S = action_new - action_old
        if r_random < np.exp(-delta_S):
            x[bead] = x_new

        # Wait until thermalization
        if step > 0.15 * n_steps:
            path += [x.copy()]
    return path

if __name__ == "__main__":
    
    N = 100      # Number of time slices
    n_steps = 1000000  # Number of Monte Carlo steps
    beta = 10    # Inverse temperature
    dt = beta / N # Time step
    mw = 1.0      # m omega of oscillator
    m  = 1.0      # mass of oscillator
    limit = 0.6   # Limit for position updates

    path = monteCarloMethod(N, n_steps, beta, dt, limit=limit, mw=mw, m=m)

    import matplotlib.pyplot as plt
    hist = np.concatenate(path)

    x_vals = np.linspace(-3, 3, 200)
    prob_exact = np.sqrt(mw/np.pi*np.tanh(beta*mw/2)) * np.exp(-mw*np.tanh(beta*mw/2)*x_vals**2)
    plt.plot(x_vals, prob_exact, 'r-', label="$P_{exact}(x, \\beta, m, \\omega)$")
    plt.hist(hist, bins=50, density=True, alpha=0.5, label="PIMC")
    plt.legend()
    plt.title(f"Path Integral Monte Carlo: Quadratic Potential ($\\beta={beta}$, $m\\omega^2={mw}$)")
    plt.grid()
    plt.show()
