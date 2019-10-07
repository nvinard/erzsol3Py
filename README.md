# pythonTools4Erzsol3
A few useful Python function to easily create input files used for erzsol3 and convert erzsol3s binary output to an hdf5 database.
The erzsol3 output is written to hdf5 here with some additional beta information that will be used in a machine learning application.

Erzsol3 is a Fortran code written by Brian Kennet. The original version can be downloaded at http://www.quest-itn.org/library/software/reflectivity-method.html
Two files have been slighly changed here. In qbessel.f line 136 was changed from COMPLEX FUNCTION BESHS0*16(X, IFAIL) to  COMPLEX*16 FUNCTION BESHS0(X, IFAIL) in order to overcome an error occuring during compiling. A copy of erzol3.f named erzsol3c.f was made and it was changed to allow a greater range of slownesses and frequencies from 600 and 2500 to 10000 and 3600, respectively.

The fortran code is compiled in the following way form the directory of this README file
gfortran -mcmodel=medium -O2 -o erzsol3/bin/erzsol3 erzsol3/src/erzsol3c.f erzsol3/src/qbessel.f erzsol3/src/qfcoolr.f
