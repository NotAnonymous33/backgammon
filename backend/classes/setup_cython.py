# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension(
        "CBoard",
        ["CBoard.pyx"],
    ),
    Extension(
        "agents.CMCTS",
        ["agents/CMCTS.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}, annotate=True),
    packages=find_packages()
)