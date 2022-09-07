from setuptools import setup, find_packages

setup(
    name="brainspy-smg",
    version="1.0.0",
    description=
    "Automatisation for creating neural-network based surrogate models of boron-doped silicon chips.",
    url="https://github.com/BraiNEdarwin/brainspy-smg",
    author=
    "This has adopted part of the BRAINS skynet repository code, which has been cleaned and refactored. The maintainers of the code are Hans-Christian Ruiz Euler and Unai Alegre Ibarra.",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    packages=find_packages(),
    zip_safe=False,
)
