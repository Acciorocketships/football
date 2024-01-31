from setuptools import find_packages, setup

setup(
    name="football",
    version="0.0.1",
    description="VMAS Football with Benchmarl",
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    packages=find_packages(),
    install_requires=["torch", "torchrl", "moviepy", "wandb"],
)
