#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
namespace deepf1_math_utils {
	std::vector<float> polyvalf(const std::vector<float>& oCoeff,
		const std::vector<float>& oX);

	/**
	
	*/
	std::vector<float> polyfitf(const std::vector<float>& oX,
		const std::vector<float>& oY, int nDegree);

	std::vector<double> polyvald(const std::vector<double>& oCoeff,
		const std::vector<double>& oX);
	std::vector<double> polyfitd(const std::vector<double>& oX,
		const std::vector<double>& oY, int nDegree);
}