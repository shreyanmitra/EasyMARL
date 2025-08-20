#!/usr/bin/env python3
"""
Setup script for EasyMARL - Educational Mu    name="easymarl",
    version="1.0.0",
    author="Shreyan Mitra",
    author_email="shreyan.m.mitra@gmail.com",
    description="Educational Multi-Agent Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyanmitra/EasyMARL",
    project_urls={
        "Bug Tracker": "https://github.com/shreyanmitra/EasyMARL/issues",
        "Documentation": "https://github.com/shreyanmitra/EasyMARL#readme",
        "Source Code": "https://github.com/shreyanmitra/EasyMARL",
        "Demo": "https://shreyanmitra.github.io/EasyMARL"
    },forcement Learning Framework

This script creates a pip-installable package that provides:
1. Easy-to-use MARL algorithms for education and research
2. Professional web-based GUI interface
3. 20+ implemented MARL algorithms
4. World-class performance optimizations
5. Comprehensive documentation and tutorials
"""

from setuptools import setup, find_packages
import os
import sys

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle version specifications
                if ">=" in line:
                    requirements.append(line)
                elif "==" in line:
                    requirements.append(line)
                else:
                    requirements.append(line)
    return requirements

# Optional dependencies for enhanced features
extras_require = {
    "enhanced": [
        "numba>=0.57.0",
        "jax>=0.4.0", 
        "jaxlib>=0.4.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0"
    ],
    "gui": [
        "gradio>=4.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0"
    ],
    "tracking": [
        "wandb>=0.12.21"
    ],
    "video": [
        "moviepy>=1.0.3",
        "ffmpeg>=1.4"
    ],
    "optimization": [
        "cvxpy>=1.2.0",
        "networkx>=2.6.0"
    ]
}

# All optional dependencies
extras_require["all"] = [
    dep for deps_list in extras_require.values() 
    for dep in deps_list
]

setup(
    name="easymarl",
    version="1.0.0",
    author="Shreyan Mitra",
    author_email="shreyan.m.mitra@gmail.com",
    description="Educational Multi-Agent Reinforcement Learning Framework with 20+ Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyanmitra/EasyMARL",
    project_urls={
        "Bug Tracker": "https://github.com/shreyanmitra/EasyMARL/issues",
        "Documentation": "https://github.com/shreyanmitra/EasyMARL#readme",
        "Source Code": "https://github.com/shreyanmitra/EasyMARL",
        "Demo": "https://shreyanmitra.github.io/EasyMARL"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "easymarl": [
            "config/*.yaml",
            "config/**/*.yaml", 
            "examples/*.py",
            "gui.py",
            "react-frontend/build/**/*",
            "react-frontend/public/**/*"
        ]
    },
    entry_points={
        "console_scripts": [
            "easymarl-gui=easymarl:launch_gui",
            "easymarl-train=easymarl.main:cli_main",
            "easymarl-demo=easymarl.examples.demo:main"
        ]
    },
    keywords=[
        "multi-agent", "reinforcement-learning", "machine-learning", 
        "artificial-intelligence", "education", "research", "marl",
        "q-learning", "policy-gradient", "actor-critic", "cooperative-ai"
    ],
    zip_safe=False,
)
