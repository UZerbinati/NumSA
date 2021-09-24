#ifndef MESHREF
#define MESHREF
#include <Eigen/Dense>
using Eigen::MatrixXd;
#include <Eigen/StdVector>
class Mesh
{
	private:
		//Mesh dimension
		int dim; 
		//Cells list
		std::vector <std::vector<int>> cells;
		//Points in the mesh
		std::vector < MatrixXd > points;
	public:
		Mesh();
		Mesh(int d);
		void SetCells(std::vector<std::vector<int>> cs);
		void SetPoints(std::vector< MatrixXd > ps);
		std::string disp();

};
Mesh UniformMesh(MatrixXd a, MatrixXd b, double h);
#endif
