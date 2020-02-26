import abc

import numpy as np
from scipy.interpolate import barycentric_interpolate
from scipy.integrate._ode import IntegratorBase


class GaussIntegrator(IntegratorBase):
    """Gaussian Quadrature

    Simple implementation of a symplectic Gauss integrator,
    order 4 and 6 (s=2 and s=3).

    Note, this is a modified code recieved from Simen Kvaal and
    Thomas Bondo Pedersen.

    Parameters
    ----------
    s : int
        order = 2 * s
    maxit : int
        Maximum number of iterations
    eps : float
        Tolerance parameter; e.g. 1e-8
    """

    supports_step = 1

    def __init__(self, s=3, maxit=20, eps=1e-6):
        assert maxit > 0

        self.s = s
        self.maxit = maxit
        self.eps = eps

        self.y = None
        self.y_prev = None

        self.a, self.b, self.c = gauss_tableau(self.s, np=np)

        self.rhs_evals = 0
        self.success = 1

    def eval_rhs(self, y, t):
        self.rhs_evals += 1

        return self.rhs(y, t)

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
            F = np.zeros((self.n, self.s), dtype=np.complex128)
            for i in range(self.s):
                F[:, i] = self.eval_rhs(y + Z[:, i], t + dt * self.c[i])

            Z_new = dt * np.matmul(F, self.a.transpose())
            R = Z - Z_new
            Z = Z_new

            if np.linalg.norm(R) < self.eps:
                converged = True
                break

        if not converged:
            self.success = 0

        return Z, F

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):

        # Predict solution Z of nonlinear equations
        # Note that a fixed 8th order predictor is implemented

        # Compute interpolating polynomial w(t), that interpolates (t_{n-1},
        # y_{n-1}) and the points (t_{n-1}+c_i*h,Y_{n-1,i}).

        self.rhs = f
        u = y0
        dt = t1 - t0
        t = t0

        self.y = u.astype(np.complex128)
        self.n = len(self.y)

        if self.y_prev is None:
            self.y_prev = np.zeros_like(self.y)
            self.y_prev += self.y
            self.Z = np.zeros((1, self.n, self.s), dtype=np.complex128)

        t_vec = (t - dt) + np.append([0], dt * self.c)
        t_vec2 = t + dt * self.c

        W = np.zeros((self.n, self.s + 1), dtype=np.complex128)
        W[:, 0] += self.y_prev

        for i in range(self.s):
            W[:, i + 1] = self.y_prev + self.Z[0, :, i]

        Y0 = barycentric_interpolate(t_vec, W.transpose(), t_vec2).transpose()

        # Save as initial guess Z0
        Z0 = np.zeros((self.n, self.s), dtype=np.complex128)
        for i in range(self.s):
            Z0[:, i] = Y0[:, i] - self.y

        # Solve nonlinear equations
        Z_new, self.F = self.Z_solve(self.y, Z0, t, dt)

        # Store solution for next predictor step
        self.Z[1:, :, :] = self.Z[:-1, :, :]
        self.Z[0, :, :] = np.array(Z_new)

        # save previous vector
        self.y_prev = np.array(self.y)

        # Make a step using Gauss method
        for i in range(self.s):
            self.y += dt * self.b[i] * self.F[:, i]

        return (self.y, t1)

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