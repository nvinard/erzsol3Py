import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erzsol3Py",
    version="0.1",
    author="Nicolas Vinard",
    author_email="vinard.nicolas@gmail.com",
    description="Some functions to create erzsol3 input files, read output files and a little more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvinard/erzsol3Py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
