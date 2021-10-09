#include "./svd.hpp"
#include <vector>
#include <string>
#include <cassert>

SVD(std::function <VectorXd(VectorXd)> L){
	action = L;
}

void SVDBind(py::module &module){
	py::class_<SVD>(module, "SVD")
		.def(py::init< std::function <VectorXd(VectorXd)>  >());
}

