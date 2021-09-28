#include "./space.hpp"
#include <vector>
#include <string>
#include <cassert>
void Space::pushBase(std::function<MatrixXd(MatrixXd)> phi){
	Basis.push_back(phi);
}
void Space::SetMesh(Mesh m){
	mesh = m;
}
CInf::CInf(Mesh m){
	int pnts_number;	
	std::vector<MatrixXd> MeshPoints;
	SetMesh(m);
	pnts_number = mesh.PointsNumber();

	for (int i=0; i < pnts_number; i++){
		//We define the Kroneker delta that form the basis of the discrete space
		std::function <MatrixXd(MatrixXd)> delta_i = [&MeshPoints,i](MatrixXd x){
			MatrixXd v(1,1);
			v << 0.0;
			if(x == MeshPoints[i]){
			       v << 1.0;
			}
			return v;
		};
		pushBase(delta_i);
	}

}
std::string CInf::disp(){
	return "CInf is space used to approximate smooth function using finite differences schemes.";
}
MatrixXd CInf::GetValue(MatrixXd Dofs, MatrixXd x){
	assert(Dofs.rows() == Basis.size() && "More degree of freedom rather then base function, not possinle in the space CInf.");
	MatrixXd value;
	value = Basis[0](x);
	for (int i=0; i < Dofs.rows(); i++){
		value = value + Dofs(i,0)*Basis[i](x); 
	}
	return value;
}
//Dummy declaration because if they are not defined PyBind11 will give an error.
double CInf::Quadrature(MatrixXd Dofs){ return 0.0;}
MatrixXd CInf::ReferenceMap(MatrixXd Dofs){ return MatrixXd::Random(1,1); };
void SpaceBind(py::module &module){
	py::class_<CInf>(module, "CInf")
		.def(py::init<Mesh>())
		.def("__repr__", &CInf::disp)
		.def("Quadrature", &CInf::Quadrature)
		.def("GetValue", &CInf::GetValue)
		.def("ReferenceMap", &CInf::ReferenceMap);
}
