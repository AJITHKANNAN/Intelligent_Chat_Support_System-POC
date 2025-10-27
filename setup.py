from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ICSS-RAG setup",
    version="0.1.0",
    author="Ajith",
    packages=find_packages(),
    install_requires = requirements,
)