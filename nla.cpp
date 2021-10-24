/*
NumSA
Umberto Zerbinati Copyright 2021
Developed at KAUST, Saudi Arabia
*/

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "la/svd.cpp"

