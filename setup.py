"""setuptools config script"""
from setuptools import setup, find_packages

setup(
    name="ml_rest_fastapi",
    version="0.1.0",
    description="A RESTful API to return predictions from a trained ML model, \
        built with Python 3 and FastAPI",
    url="https://github.com/jgbustos/ml_rest_fastapi",
    author="Jorge Garcia de Bustos",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="python3 fastapi machine-learning rest-api",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.19.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "pydantic>=2.0",
    ],
)
