#!/bin/bash
make
pip3 install ./py[extra]
cd test
pytest .
mpiexec -n 2 pytest . --with-mpi 
