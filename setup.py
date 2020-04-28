import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssvd-pkg-JY-WZ", # Replace with your own username
    version="0.0.1",
    author="Ziang Wang & Jishen Yin",
    author_email="jishen.yin@duke.edu",
    description="A package for sparse singular value decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YinJishen/ssvd-package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)