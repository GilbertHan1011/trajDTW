from setuptools import setup, find_packages

setup(
    name="trajDTW",
    version="0.1.0",
    author="Gilbert Han",
    author_email="GilbertHan1011@gmail.com",
    description="A package for trajectory analysis and dynamic time warping for cell trajectories",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gilberthan/trajDTW",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scanpy",
        "fastdtw",
        "tqdm",
        "joblib",
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'trajdtw=trajDTW.cli:main',
        ],
    },
) 