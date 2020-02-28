import numpy as np
from scipy.integrate import complex_ode, ode
from gauss_integrator import GaussIntegrator


def rhs(t, y, arg1=1):
    return [-arg1 * y]


def test_1d_ode():
    r = ode(rhs).set_integrator("GaussIntegrator")
    r_2 = ode(rhs).set_integrator("zvode")
    r.set_initial_value([1], 0)
    r_2.set_initial_value([1], 0)

    t1 = 10
    dt = 1e-1
    eps = 1e-4

    while r.successful() and r_2.successful() and r.t < t1 and r_2.t < t1:
        g_int = r.integrate(r.t + dt)
        z_int = r_2.integrate(r_2.t + dt)
        assert abs(g_int - z_int) < eps
