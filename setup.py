from setuptools import setup, find_packages

setup(
    name="audience-intelligence",
    version="1.0.0",
    author="Nielsen AI Team",
    author_email="ai-team@nielsen.com",
    description="Cross-Modal Audience Intelligence Platform for predicting and analyzing audience engagement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nielsen/audience-intelligence",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.28.0",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pillow>=9.4.0",
        "networkx>=2.8.0",
        "spacy>=3.5.0",
        "faiss-cpu>=1.7.4",
        "onnx>=1.13.0",
        "onnxruntime>=1.14.0",
        "ray[serve]>=2.3.0",
        "jupyter>=1.0.0",
        "pytest>=7.3.1",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
            "pytest-cov",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
        "causal": [
            "dowhy>=0.9",
            "causallearn>=0.1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audience-intelligence-api=serving.api:main",
            "audience-intelligence-ray=serving.ray_serve:main",
        ],
    },
)