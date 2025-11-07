"""Setup script for adaptive-muon package."""

from setuptools import find_packages, setup

setup(
    name="adaptive-muon",
    version="0.1.0",
    description="PyTorch project with enforced code quality",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
