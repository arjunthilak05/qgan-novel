"""
Setup script for Entropy-Regularized Quantum GAN (ER-QGAN)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="er-qgan",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Entropy-Regularized Quantum Generative Adversarial Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qgan-novel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pennylane>=0.35.0",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "pennylane-lightning-gpu>=0.35.0",
        ],
        "hardware": [
            "qiskit-ibm-runtime>=0.18.0",
        ],
    },
)
