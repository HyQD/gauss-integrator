import pytest
import numpy as np
from scipy.integrate import complex_ode, ode
from gauss_integrator import GaussIntegrator


@pytest.fixture(params=["A", "B", "C"])
def method(request):
    return request.param


def rhs(t, y, arg1=1, arg2=2):
    return [-arg1 * y[0], -0.5 * arg2 * y[1]]


def test_2d_ode(method):
    r = ode(rhs).set_integrator("GaussIntegrator", method=method)
    r_2 = ode(rhs).set_integrator("zvode")
    r.set_initial_value([1, 2], 0)
    r_2.set_initial_value([1, 2], 0)

    t1 = 10
    dt = 1e-1
    eps = 1e-4

    while r.successful() and r_2.successful() and r.t < t1 and r_2.t < t1:
        g_int = r.integrate(r.t + dt)
        z_int = r_2.integrate(r_2.t + dt)
        assert np.all(abs(g_int - z_int) < eps)

    # assert False


def fo_complex_ode(t, y):
    return -1j * y


def test_fo_complex_ode(method):
    t0 = 0
    t1 = 4 * np.pi
    dt = 1e-2

    y0 = 1

    # Note that we use a very coarse comparison as the 'vode' integrator is not
    # symplectic and will perform worse compared to the `GaussIntegrator`.
    eps = 1e-4

    r = complex_ode(fo_complex_ode).set_integrator(
        "GaussIntegrator", method=method
    )
    r_2 = complex_ode(fo_complex_ode).set_integrator("vode")
    r.set_initial_value(y0, t0)
    r_2.set_initial_value(y0, t0)

    # g_int_list = []
    # z_int_list = []
    # time = []

    while r.successful() and r_2.successful() and r.t < t1 and r_2.t < t1:
        # time.append(r.t)
        # g_int_list.append(r.y)
        # z_int_list.append(r_2.y)
        g_int = r.integrate(r.t + dt)
        z_int = r_2.integrate(r_2.t + dt)
        assert np.all(abs(g_int - z_int) < eps)

    # import matplotlib.pyplot as plt

    # ax1 = plt.subplot(2, 1, 1)
    # ax1.plot(time, np.array(g_int_list).real, label="Gauss re")
    # ax1.plot(time, np.array(z_int_list).real, "--", label="Zvode re")
    # ax1.plot(time, np.array(g_int_list).imag, label="Gauss im")
    # ax1.plot(time, np.array(z_int_list).imag, "--", label="Zvode im")
    # ax1.legend()
    # ax1.grid()

    # diff = np.array(g_int_list) - np.array(z_int_list)

    # ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(time, np.abs(diff.real), label="Diff re")
    # ax2.plot(time, np.abs(diff.imag), label="Diff im")

    # ax2.legend()
    # ax2.grid()

    # plt.show()


# if __name__ == "__main__":
#     test_fo_complex_ode("A")


def ho_rhs(t, y):
    return [y[1], -y[0]]


def test_ho(method):
    t0 = 0
    t1 = 4 * np.pi
    dt = 1e-2

    y0 = [1, 0]

    # Note that we use a very coarse comparison as the 'vode' integrator is not
    # symplectic and will perform worse compared to the `GaussIntegrator`.
    eps = 1e-4

    r = complex_ode(ho_rhs).set_integrator("GaussIntegrator", method=method)
    r_2 = complex_ode(ho_rhs).set_integrator("vode")
    r.set_initial_value(y0, t0)
    r_2.set_initial_value(y0, t0)

    # assert np.all(r.y == r_2.y)

    # g_int_list = []
    # z_int_list = []
    # time = []

    while r.successful() and r_2.successful() and r.t < t1 and r_2.t < t1:
        # time.append(r.t)
        # g_int_list.append(r.y)
        # z_int_list.append(r_2.y)
        g_int = r.integrate(r.t + dt)
        z_int = r_2.integrate(r_2.t + dt)
        assert np.all(abs(g_int - z_int) < eps)

    # assert False

    # import matplotlib.pyplot as plt

    # ax1 = plt.subplot(2, 1, 1)
    # ax1.plot(time, np.array(g_int_list)[:, 0], label="Gauss x")
    # ax1.plot(time, np.array(z_int_list)[:, 0], label="Zvode x")
    # ax1.plot(time, np.array(g_int_list)[:, 1], label="Gauss y")
    # ax1.plot(time, np.array(z_int_list)[:, 1], label="Zvode y")
    # ax1.legend()
    # ax1.grid()

    # diff = np.abs(np.array(g_int_list) - np.array(z_int_list))

    # ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(time, diff[:, 0], label="Diff x")
    # ax2.plot(time, diff[:, 1], label="Diff y")
    #
    # ax2.legend()
    # ax2.grid()

    # plt.show()


# if __name__ == "__main__":
#     test_ho()
