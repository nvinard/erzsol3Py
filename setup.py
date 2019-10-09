import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erzsol3ToolBox",
    version="0.0.7",
    author="Nicolas Vinard",
    author_email="vinard.nicolas@gmail.com",
    description="Some functions to create erzsol3 input files and convert outputs to hdf5 for Machine learning using CNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvinard/pythonTools4Erzsol3/archive/0.0.7.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
