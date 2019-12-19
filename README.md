# erzsol3Py
A few useful Python functions to easily create erzsol3 input files and read output files.

Erzsol3 is a Fortran code written by Brian Kennet. The original version can be downloaded at http://www.quest-itn.org/library/software/reflectivity-method.html
Two files have been slighly changed here. In qbessel.f line 136 was changed from COMPLEX FUNCTION BESHS0*16(X, IFAIL) to  COMPLEX*16 FUNCTION BESHS0(X, IFAIL) in order to overcome an error occuring during compiling. A copy of erzol3.f named erzsol3c.f was made and it was changed to allow a greater range of slownesses and frequencies from 600 and 2500 to 10000 and 3600, respectively.

The fortran code is compiled in the following way form the directory of this README file
gfortran -mcmodel=medium -O2 -o erzsol3SourceCode/bin/erzsol3 erzsol3SourceCode/src/erzsol3c.f erzsol3SourceCode/src/qbessel.f erzsol3SourceCode/src/qfcoolr.f

erzsol3Py can be installed via pip
pip install -i https://test.pypi.org/simple/ erzsol3Py

can be imported in python as
import erzsol3Py.erzsol3Py as erz
