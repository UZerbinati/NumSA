#ifndef SPACEREF
#define SPACEREF
#include <Eigen/Dense>
using Eigen::MatrixXd;
#include <Eigen/StdVector>
class Space
{
	public:
		//Mesh we are going to build our space on
		Mesh mesh;

		//Basis function of our space
		std::vector <std::function<MatrixXd(MatrixXd)>> Basis;
		void pushBase (std::function<MatrixXd(MatrixXd)> phi); 
		void SetMesh(Mesh m);
		Mesh getMesh();
		double Quadrature(MatrixXd Dofs);
		MatrixXd GetValue(MatrixXd Dofs, MatrixXd x);  
		MatrixXd ReferenceMap(MatrixXd x);
		int nDotEl;

};

class CInf : Space{
	public:
		CInf();
		CInf(Mesh m);
		std::string disp();
		double Quadrature(MatrixXd Dofs);
		MatrixXd GetValue(MatrixXd Dofs, MatrixXd x);  
		MatrixXd ReferenceMap(MatrixXd x);
};
#endif
