#include "./mesh.hpp"
#include <vector>
#include <string>
#include <cassert>

Mesh::Mesh(){
	dim = 1;
}
Mesh::Mesh(int d){
	dim = d;
	MatrixXd m = MatrixXd::Random(3,1);
	cells = {{0,0,0}};
	points = {m};
}
void Mesh::SetCells(std::vector<std::vector<int>> cs){
	/*
	 * Function to set the points in each cell of the mesh.
	 */
	cells = cs;
}
void Mesh::SetPoints(std::vector< MatrixXd > ps){
	points = ps;
}
std::string Mesh::disp(){
	std::string msg;

	msg = "Mesh Dimension: " + std::to_string(dim) +  "\n";
	msg = msg + "Poins: \n";
	
	for (MatrixXd P: points){
		std::stringstream ss;
		ss << P;
		msg = msg + "[" + ss.str() + "]\n";
	}
	msg = msg + "Cells: \n";
	for (std::vector<int> cell: cells){
		msg = msg + "[ ";
		for (int point: cell){
			msg = msg + std::to_string(point) + " ";
		}
		msg = msg+"]\n";
	}
	return msg;
}
Mesh UniformMesh(MatrixXd a, MatrixXd b, double h){
	/*
	 * This functions create a uniform mesh
	 * from the point a, to the point b 
	 * in a uniform spaced manner.
	 */
	assert(a.rows() == b.rows() &&"Point A and B do not have the same dimensions.");
	Mesh dummy(1); // We initialize an empty mesh with dimension 1, as dummy return;
	if (a.rows() == 1){
		Mesh uniformesh(1); // We initialize an empty mesh with dimension 1.
		return uniformesh;
	}
	return dummy;

}
//Function that returns the number of elements in a mesh
int Mesh::ElNumber(){
	return cells.size();
}
//Function that returns the number of points in a cell
int Mesh::CellPointNumber(int cell_number){
	return cells[cell_number].size();
}
int Mesh::PointsNumber(){
	return points.size();
}
std::vector <MatrixXd> Mesh::GetPoints(){
	return points;
}
void MeshBind(py::module &module){
	py::class_<Mesh>(module, "Mesh")
		.def(py::init<int>())
		.def("__repr__", &Mesh::disp)
		.def("SetPoints", &Mesh::SetPoints)
		.def("SetCells", &Mesh::SetCells)
		.def("GetPoints", &Mesh::GetPoints)
		.def("ElNumber", &Mesh::ElNumber)
		.def("PointsNumber", &Mesh::ElNumber)
		.def("CellPointNumber", &Mesh::ElNumber);
}

