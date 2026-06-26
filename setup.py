from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="adapt-mllm",
    version="1.0.0",
    description="ADAPT: Attention Dynamics Alignment with Preference Tuning for Faithful MLLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zhiyuan Yao, Zheren Fu, Zhixiao Zheng, Jiajun Li, Yi Tu, Zhendong Mao",
    author_email="yaozhiyuan@mail.ustc.edu.cn",
    url="https://github.com/yao-ustc/ADAPT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "tqdm>=4.64.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
