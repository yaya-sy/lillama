# Always prefer setuptools over distutils
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / Path('README.md')) as f:
    long_description = f.read()

with open(here / Path('requirements.txt')) as f:
    requirements = f.read().split("\n")

setup(
    name='lillama',
    version='0.1.0',
    description='',
    long_description=long_description,
    url='https://github.com/yaya-sy/lillama',
    author='Yaya Sy',
    author_email='yayasysco@gmail.com',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9'
    ],
    keywords='',
    entry_points={'console_scripts': ['lillama-data = lillama.src.data:main']},
    packages=find_packages(exclude=['docs', 'test', "checkpoints", "sscorer"]),
    setup_requires=['setuptools>=38.6.0'],  # >38.6.0 needed for markdown README.md
    install_requires=requirements
)