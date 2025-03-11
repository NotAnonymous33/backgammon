from setuptools import setup, Extension
import platform
import sys

# Get information about the Python environment
is_64bits = sys.maxsize > 2**32
python_version = platform.python_version()

# Set compiler flags based on architecture
extra_compile_args = []
extra_link_args = []

# with optimizations 1000 random vs random 11.49 11.71 11.23
# without optimizations 1000 random vs random 11.08 10.76 10.76

if sys.platform == 'win32':
    if is_64bits:
        extra_compile_args = ['/std:c++17', '/EHsc', '/MD', '/DWIN64']#, '/Ox']
    else:
        extra_compile_args = ['/std:c++17', '/EHsc', '/MD', '/DWIN32']#, '/Ox']
else:
    extra_compile_args = ['-std=c++17']

# Configure extension
ext_modules = [
    Extension(
        'board_cpp',
        ['board_cpp.cpp'],
        include_dirs=[],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# Try to add pybind11 includes
try:
    import pybind11
    for ext in ext_modules:
        ext.include_dirs.append(pybind11.get_include())
        ext.include_dirs.append(pybind11.get_include(user=True))
    setup_requires = []
except ImportError:
    setup_requires = ['pybind11>=2.6.0']

setup(
    name='board_cpp',
    version='0.1',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    setup_requires=setup_requires,
    zip_safe=False,
    python_requires='>=3.7',
)