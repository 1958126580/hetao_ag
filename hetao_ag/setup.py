# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="hetao_ag",
    version="1.0.0",
    author="Hetao College",
    author_email="hetao@example.com",
    description="河套智慧农牧业库 - Smart Agriculture & Animal Husbandry Library",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/hetao-college/hetao_ag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "full": [
            "pyyaml>=6.0",
            "scipy>=1.10",
            "pandas>=2.0",
        ],
        "space": [
            "rasterio>=1.3",
            "geopandas>=0.13",
        ],
        "livestock": [
            "torch>=2.0",
            "opencv-python>=4.8",
            "ultralytics>=8.0",
        ],
        "opt": [
            "pulp>=2.7",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hetao-demo=examples.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
