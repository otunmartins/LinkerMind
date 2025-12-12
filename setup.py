from setuptools import setup, find_packages

setup(
    name="linkermind",
    version="1.0.0",
    description="Mechanism-informed deep learning framework for ADC linker design",
    author="LinkerMind Team",
    author_email="otunmartins@outlook.com",
    packages=find_packages(),
    install_requires=[
        "rdkit>=2022.09.1",
        "torch>=1.12.1",
        "scikit-learn>=1.1.2",
        "pandas>=1.5.0",
    ],
    python_requires=">=3.8",
)
