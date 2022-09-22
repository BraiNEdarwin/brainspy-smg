from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="brainspy-smg",
    version="1.0.7",
    description=
    "A python package to support research on different nano-scale materials for creating hardware accelerators in the context of deep neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['pytorch', 'nano', 'material', 'hardware acceleration', 'deep neural networks', 'surrogate model', 'twin device'],	
    author="Unai Alegre-Ibarra et al.",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    #python_requires='==3.9',
    install_requires=["brainspy", "scipy~=1.5", "more_itertools==8.6.0"],
    packages=find_packages(exclude=["tests"]),
    url="https://github.com/BraiNEdarwin/brainspy-smg",
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ]
)
