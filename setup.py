from setuptools import setup, find_packages

setup(
    name="adapt",
    version="1.0.0",
    description="ADAPT: Attention Dynamics Alignment with Preference Tuning for Faithful MLLMs",
    author="Zhiyuan Yao et al.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "Pillow",
        "numpy",
        "opencv-python",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
