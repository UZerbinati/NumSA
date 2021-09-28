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

#include "core/mesh.cpp"
#include "core/space.cpp"

PYBIND11_MODULE(core, module) {
    module.attr("release") = "0.0.1";
    MeshBind(module);
    SpaceBind(module);
}
