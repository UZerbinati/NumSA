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
		int getDim();
		void SetCells(std::vector<std::vector<int>> cs);
		void SetPoints(std::vector< MatrixXd > ps);
		std::string disp();
		int ElNumber();
		int CellPointNumber(int cell_number);
		int PointsNumber();
		std::vector <MatrixXd> GetPoints();

};
Mesh UniformMesh(MatrixXd a, MatrixXd b, double h);
#endif
