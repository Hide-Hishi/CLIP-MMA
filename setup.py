from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="clip-mma",
    version="1.0",
    description="",
    author="Hidekazu Hishinuma",
    url="https://github.com/Hide-Hishi/CLIP-MMA.git",
    packages=find_packages(),
    py_modules=[],
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=_requires_from_file('requirements.txt'),
)
