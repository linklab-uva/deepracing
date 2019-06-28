#ifndef F1_DATALOGGER_EIGEN_UTILS_H
#define F1_DATALOGGER_EIGEN_UTILS_H
#include <Eigen/Core>
#include <vector>
namespace deepf1
{
	class EigenUtils
	{
	public:
		EigenUtils();
		~EigenUtils();
		static std::vector < Eigen::Vector4d > loadTrackFile(const std::string& trackfile, const double& interpolation_factor = -1.0, bool debug = false);
		static Eigen::MatrixXd vectorToMatrix(const std::vector < Eigen::Vector4d >& vector);
	};

}
#endif