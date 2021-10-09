#ifndef SVDREF
#define SVDREF
#include <Eigen/Dense>
#include <Eigen/StdVector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
class SVD
{
	private:
		std::function <VectorXd(VectorXd)> action;
	public:
		SVD(std::function <VectorXd(VectorXd)> L);
};
#endif
