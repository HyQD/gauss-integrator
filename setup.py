from setuptools import setup, find_packages

setup(
    name="gauss-integrator",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["numpy", "scipy",],
)
