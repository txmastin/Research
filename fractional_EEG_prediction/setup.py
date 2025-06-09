from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="flifnetwork",
    ext_modules=cythonize("flifnetwork.pyx", language_level=3),
    include_dirs=[np.get_include()]
)

