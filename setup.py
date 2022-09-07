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
    install_requires=[
        'scipy~=1.5.4','more_itertools==8.6.0'
    ],
    zip_safe=False,
)
