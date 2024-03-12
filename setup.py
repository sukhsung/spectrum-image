from setuptools import setup, find_packages

setup(
name="eels",
version="0.1.0",
description="Python package for reproducible EELS (Electron Energy Loss Spectroscopy) analysis",
author="Suk Hyun Sung",
packages=find_packages(),
classifiers=[
"Programming Language :: Python :: 3",
"License :: OSI Approved :: MIT License",
"Operating System :: OS Independent",
],
python_requires=">=3.8",
)