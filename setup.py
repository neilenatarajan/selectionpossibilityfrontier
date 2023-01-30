from setuptools import setup, find_packages

setup(
  name='selectionpossibilityfrontier',
  version='0.0.2',
  author='Neil Natarajan',
  author_email='neilenatarajan@gmail.com',
  packages=find_packages(include=['selectionpossibilityfrontier', 'selectionpossibilityfrontier.*']),
  install_requires=[
    'pandas>=0.18',
    'numpy>=1.10',
    'seaborn>=0.6.0',
    'matplotlib>=1.4.3',
    'future',
  ],
)