from setuptools import setup, find_packages

setup(
    setup_requires=['pbr>=1.9', 'setuptools>=17.1'],
    pbr=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
