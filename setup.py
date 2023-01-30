from setuptools import setup, findpackages

setup(
  name='frontier',
  version='0.0.1',
  author='Neil Natarajan',
  author_email='neilenatarajan@gmail.com',
  packages=find_packages(include=['selectionpossibilityfrontier', 'selectionpossibilityfrontier.*'],
  install_requires=[
    'pandas==0.23.3',
    'numpy>=1.14.5',
    'matplotlib>=2.2.0'
  ]
)