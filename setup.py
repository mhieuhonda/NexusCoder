from setuptools import setup, find_packages

setup(
    name="nexus-coder",
    version="0.1.0",
    description="Nexus Coder - AI Agent với kiến trúc MoE 10B/1.5B active",
    author="Hieu Louis",
    author_email="mhieuhonda@users.noreply.github.com",
    url="https://github.com/mhieuhonda/NexusCoder",
    packages=find_packages(),
    python_requires="==3.12.*",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
