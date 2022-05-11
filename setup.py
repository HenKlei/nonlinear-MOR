from setuptools import setup, find_packages


dependencies = [
    'numpy',
    'geodesic_shooting @ git+https://github.com/HenKlei/geodesic-shooting.git',
    'tent_pitching @ git+https://github.com/HenKlei/tent-pitching.git',
]

setup(
    name='nonlinear-mor',
    version='0.1.0',
    description='Algorithms for nonlinear model order reduction',
    author='Hendrik Kleikamp',
    maintainer='Hendrik Kleikamp',
    maintainer_email='hendrik.kleikamp@uni-muenster.de',
    packages=find_packages(),
    install_requires=dependencies,
)
