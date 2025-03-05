# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "CBoard",
        ["CBoard.pyx"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)