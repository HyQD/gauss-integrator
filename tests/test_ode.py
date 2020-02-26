import numpy as np
from scipy.integrate import complex_ode, ode
from gauss_integrator import GaussIntegrator


def rhs(t, y, arg1=1):
    return [-arg1 * y]


def test_1d_ode():
    r = ode(rhs).set_integrator("GaussIntegrator")
    r = ode(rhs).set_integrator("zvode")
    # r = complex_ode(rhs).set_integrator("GaussIntegrator")
    r.set_initial_value([1], 0)

    t1 = 10
    dt = 1e-1

    while r.successful() and r.t < t1:
        print(r.t + dt, r.integrate(r.t + dt))

    assert False
