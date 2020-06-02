import abc

import numpy as np
from scipy.interpolate import barycentric_interpolate
from scipy.integrate._ode import IntegratorBase


class GaussIntegrator(IntegratorBase):
    """Gaussian Quadrature

    Simple implementation of a symplectic Gauss integrator, any order.

    Note, this is a modified code recieved from Simen Kvaal and Thomas Bondo
    Pedersen.

    Parameters
    ----------
    s : int
        Integrator order. Order is defind as ``order = 2 * s``. Default is
        ``3`` yielding an integrator of order ``6``.
    maxit : int
        The maximum number of fix-point iterations. Default is ``20``.
    eps : float
        Tolerance parameter for convergence of the fix-point iterations.
        Default is ``1e-6``.
    method : str
        Solver method to use. Valid options are ``"A"``, ``"B"`` and ``"C"``,
        with ``"A"`` as the default.
    mu : float
        Parameter to method ``"B"``.
    """

    supports_step = 1

    def __init__(self, s=3, maxit=20, eps=1e-6, method="A", mu=1.75):
        assert maxit > 0
        assert method.upper() in ["A", "B", "C"]

        self.s = s
        self.maxit = maxit
        self.eps = eps

        self.a, self.b, self.c = gauss_tableau(self.s, np=np)

        SCHEMES = dict(
            A=(self._reset_A, self._run_A),
            B=(self._reset_B, self._run_B),
            C=(self._reset_C, self._run_C),
        )
        self.reset, self.run = SCHEMES[method.upper()]

        if method.upper() == "B":
            self.mu = mu

        self.rhs_evals = 0
        self.success = 1

    def reset_dec(reset_func):
        def new_reset(self, n, has_jac):
            self.n = n
            self.y_prev = np.zeros(self.n)
            reset_func(self, n, has_jac)
            self.success = 1

        return new_reset

    @reset_dec
    def _reset_A(self, n, has_jac):
        self.Z = np.zeros((1, self.n, self.s))

    @reset_dec
    def _reset_B(self, n, has_jac):
        self.F = np.zeros((self.n, self.s))
        self.Z = np.zeros((1, self.n, self.s))

        # compute mu from (6.11) in HLW
        A = np.zeros((self.s, self.s))
        u = np.zeros((self.s,))
        for j in range(self.s):
            u[j] = self.mu ** (j + 1) / (j + 1)
            for k in range(self.s):
                A[j, k] = self.c[j] ** k

        self.mu_mat = u @ np.linalg.inv(A)

        assert np.isclose(
            self.mu, np.sum(self.mu_mat)
        ), "sum(mu_mat) and mu are not close!"

        # These are used in method B for starting guesses.  We compute them
        # from Eq. (6.9) in HLW, written as a linear system [beta; nu] A = U
        self.alpha = np.zeros((self.s, self.s))
        self.nu = np.zeros((self.s,))

        U = np.zeros((self.s, self.s + 1))
        for i in range(self.s):
            for k in range(self.s):
                U[i, k] = (1 + self.c[i]) ** (k + 1) / (k + 1)
        U[:, self.s] = self.b @ (self.c ** self.s) + self.a @ (
            (self.c + 1) ** self.s
        )

        A = np.zeros((self.s + 1, self.s + 1))
        A[self.s, :] = self.mu ** np.arange(self.s + 1)
        for j in range(self.s):
            for k in range(self.s + 1):
                A[j, k] = self.c[j] ** k

        C = U @ np.linalg.inv(A)
        beta = np.array(C[:, : self.s])
        for j in range(self.s):
            self.alpha[:, j] = beta[:, j] - self.b[j]
        self.nu = np.array(C[:, self.s])

    @reset_dec
    def _reset_C(self, n, has_jac):
        # Method C uses by default 8 previous points
        self.Z = np.zeros((8, self.n, self.s))

    def eval_rhs(self, t, y):
        self.rhs_evals += 1

        return np.array(self.rhs(t, y))

    def Z_solve(self, y, Z0, t, dt):
        """Method solving the problem

        .. math:: Z = hf(y + Z)a^T

        by fix point iterations.

        Parameters
        ----------
        y : np.array
            Flattened array of amplitudes ??
        Z0 : np.array
            Initial guess
        t : float
            Current time step
        dt : float
            Time step length

        Returns
        -------
        tuple
            The converged solution :math:`Z` and the right-hand side evaluations
            :math:`F`.
        """

        Z = Z0
        converged = False

        for j in range(self.maxit):
            F = np.zeros((self.n, self.s), dtype=y.dtype)
            for i in range(self.s):
                F[:, i] = self.eval_rhs(t + dt * self.c[i], y + Z[:, i])

            Z_new = dt * np.matmul(F, self.a.transpose())
            R = Z - Z_new
            Z = Z_new

            if np.linalg.norm(R) < self.eps:
                converged = True
                break

        if not converged:
            self.success = 0

        return Z, F

    def run_dec(run_func):
        def new_run(self, f, jac, y0, t0, t1, f_params, jac_params):
            self.rhs = f
            dt = t1 - t0
            self.y = y0.copy()

            Z0 = run_func(self, self.y, t0, dt)

            # Solve nonlinear equations
            Z_new, self.F = self.Z_solve(self.y, Z0, t0, dt)

            # Store solution for next predictor step
            self.Z[1:, :, :] = self.Z[:-1, :, :]
            self.Z[0, :, :] = np.array(Z_new)

            # save previous vector
            self.y_prev = self.y.copy()

            # Make a step using Gauss method
            for i in range(self.s):
                self.y += dt * self.b[i] * self.F[:, i]

            return self.y, t1

        return new_run

    @run_dec
    def _run_A(self, y0, t0, dt):

        # Predict solution Z of nonlinear equations
        # Note that a fixed 8th order predictor is implemented

        # Compute interpolating polynomial w(t), that interpolates (t_{n-1},
        # y_{n-1}) and the points (t_{n-1}+c_i*h,Y_{n-1,i}).

        t_vec = (t0 - dt) + np.append([0], dt * self.c)
        t_vec2 = t0 + dt * self.c

        W = np.zeros((self.n, self.s + 1))
        W[:, 0] += self.y_prev

        for i in range(self.s):
            W[:, i + 1] = self.y_prev + self.Z[0, :, i]

        Y0 = barycentric_interpolate(t_vec, W.transpose(), t_vec2).transpose()

        # Save as initial guess Z0
        Z0 = np.zeros((self.n, self.s))
        for i in range(self.s):
            Z0[:, i] = Y0[:, i] - y0

        return Z0

    @run_dec
    def _run_B(self, y0, t0, dt):
        # order s+1 type method from HLW.

        # Extra stage vector computed from previous step
        Z_extra = np.zeros((self.n,))

        for i in range(self.s):
            Z_extra += dt * self.mu_mat[i] * self.F[:, i]

        # Extra rhs evauation
        # IS THE TIME RIGHT?
        F_extra = self.eval_rhs(t0 - dt + dt * self.mu, self.y_prev + Z_extra)

        Z0 = dt * np.array(self.F @ self.alpha.transpose())

        for i in range(self.s):
            Z0[:, i] += dt * self.nu[i] * F_extra

        return Z0

    @run_dec
    def _run_C(self, y0, t0, dt):
        DZ1 = self.Z[:-1, :, :] - self.Z[1:, :, :]
        DZ2 = DZ1[:-1, :, :] - DZ1[1:, :, :]
        DZ3 = DZ2[:-1, :, :] - DZ2[1:, :, :]
        DZ4 = DZ3[:-1, :, :] - DZ3[1:, :, :]
        DZ5 = DZ4[:-1, :, :] - DZ4[1:, :, :]
        DZ6 = DZ5[:-1, :, :] - DZ5[1:, :, :]
        DZ7 = DZ6[:-1, :, :] - DZ6[1:, :, :]
        Z0 = (
            self.Z[0, :, :]
            + DZ1[0, :, :]
            + DZ2[0, :, :]
            + DZ3[0, :, :]
            + DZ4[0, :, :]
            + DZ5[0, :, :]
            + DZ6[0, :, :]
            + DZ7[0, :, :]
        )

        return Z0

    def step(self, *args):
        return self.run(*args)


if GaussIntegrator not in IntegratorBase.integrator_classes:
    IntegratorBase.integrator_classes.append(GaussIntegrator)


def gauleg(n, np):
    """Compute weights and abscissa for Gauss-Legendre quadrature.

    Adapted from an old MATLAB code from 2011 by S. Kvaal.

    Usage: x,w = gauleg(n)

    Uses Golub-Welsh algorithm (Golub & Welsh, Mathematics of Computation, Vol.
    23, No. 106, (Apr., 1969), pp. 221-230)

    In the algorithm, one computes a tridiagonal matrix T, whose elements are
    given by the recurrence coefficients of the orthogonal polynomials one
    wishes to compute Gauss-quadrature rules from.  Thus, gauleg is easily
    adaptable to other orthogonal polynomials.
    """
    nn = np.arange(1, n + 1)

    a = np.sqrt((2 * nn - 1) * (2 * nn + 1)) / nn
    b = 0 * nn
    temp = (2 * nn + 1) / (2 * nn - 3)
    temp = (
        np.abs(temp) + temp
    ) / 2  # hack to remove negative indices, which are not
    # used but still give a runtime warning in np.sqrt
    c = (nn - 1) / nn * np.sqrt(temp)

    alpha = -b / a
    beta = np.sqrt(c[1:] / (a[:-1] * a[1:]))

    mu0 = 2

    J = np.diag(beta, -1) + np.diag(alpha) + np.diag(beta, 1)
    v, u = np.linalg.eig(J)
    j = np.argsort(v)
    w = mu0 * u[0, :] ** 2
    w = w[j]
    x = v[j]

    return x, w


def lagpol(c, j, np):
    """Compute Lagrange interpolation polynomial.

    Usage:  p = lagpol(c,j)

    Given a vector of collocation points c[i], compute the j'th
    Lagrange interpolation polynomial.

    Returns a np.poly1d object.
    """

    r = np.delete(c, j)
    a = np.prod(c[j] - r)
    p = np.poly1d(r, r=True) / a

    return p


def gauss_tableau(s, np):
    """Compute Butcher Tableau of s-stage Gauss integrator.

    Usage a,b,c = gauss_tableau(s)
    """

    # compute collocation points and weights
    c, b = gauleg(s, np=np)
    c = (c + 1) / 2
    b = b / 2

    # compute a matrix
    a = np.zeros((s, s))
    for j in range(s):
        p = np.polyint(lagpol(c, j, np=np))
        for i in range(s):
            a[i, j] = np.polyval(p, c[i]) - np.polyval(p, 0)

    return a, b, c
