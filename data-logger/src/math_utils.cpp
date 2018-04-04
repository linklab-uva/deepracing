#pragma once
#include "math_utils.h" 

namespace deepf1_math_utils {
	std::vector<float> polyvalf(const std::vector<float>& oCoeff,
		const std::vector<float>& oX)
	{
		size_t nCount = oX.size();
		size_t nDegree = oCoeff.size();
		std::vector<float>	oY(nCount);

		for (size_t i = 0; i < nCount; i++)
		{
			float nY = 0;
			float nXT = 1;
			float nX = oX[i];
			for (size_t j = 0; j < nDegree; j++)
			{
				// multiply current x by a coefficient
				nY += oCoeff[j] * nXT;
				// power up the X
				nXT *= nX;
			}
			oY[i] = nY;
		}

		return oY;
	}
	std::vector<float> polyfitf(const std::vector<float>& oX,
		const std::vector<float>& oY, int nDegree)
	{
		using namespace boost::numeric::ublas;

		if (oX.size() != oY.size())
			throw std::invalid_argument("X and Y vector sizes do not match");

		// more intuative this way
		nDegree++;

		size_t nCount = oX.size();
		matrix<float> oXMatrix(nCount, nDegree);
		matrix<float> oYMatrix(nCount, 1);

		// copy y matrix
		for (size_t i = 0; i < nCount; i++)
		{
			oYMatrix(i, 0) = oY[i];
		}

		// create the X matrix
		for (size_t nRow = 0; nRow < nCount; nRow++)
		{
			float nVal = 1.0f;
			for (int nCol = 0; nCol < nDegree; nCol++)
			{
				oXMatrix(nRow, nCol) = nVal;
				nVal *= oX[nRow];
			}
		}

		// transpose X matrix
		matrix<float> oXtMatrix(trans(oXMatrix));
		// multiply transposed X matrix with X matrix
		matrix<float> oXtXMatrix(prec_prod(oXtMatrix, oXMatrix));
		// multiply transposed X matrix with Y matrix
		matrix<float> oXtYMatrix(prec_prod(oXtMatrix, oYMatrix));

		// lu decomposition
		permutation_matrix<int> pert(oXtXMatrix.size1());
		const std::size_t singular = lu_factorize(oXtXMatrix, pert);
		// must be singular
		BOOST_ASSERT(singular == 0);

		// backsubstitution
		lu_substitute(oXtXMatrix, pert, oXtYMatrix);

		// copy the result to coeff
		return std::vector<float>(oXtYMatrix.data().begin(), oXtYMatrix.data().end());
	}

	std::vector<double> polyvald(const std::vector<double>& oCoeff,
		const std::vector<double>& oX) {
		size_t nCount = oX.size();
		size_t nDegree = oCoeff.size();
		std::vector<double>	oY(nCount);

		for (size_t i = 0; i < nCount; i++)
		{
			double nY = 0;
			double nXT = 1;
			double nX = oX[i];
			for (size_t j = 0; j < nDegree; j++)
			{
				// multiply current x by a coefficient
				nY += oCoeff[j] * nXT;
				// power up the X
				nXT *= nX;
			}
			oY[i] = nY;
		}

		return oY;
	}
	std::vector<double> polyfitd(const std::vector<double>& oX,
		const std::vector<double>& oY, int nDegree) {
		using namespace boost::numeric::ublas;

		if (oX.size() != oY.size())
			throw std::invalid_argument("X and Y vector sizes do not match");

		// more intuative this way
		nDegree++;

		size_t nCount = oX.size();
		matrix<double> oXMatrix(nCount, nDegree);
		matrix<double> oYMatrix(nCount, 1);

		// copy y matrix
		for (size_t i = 0; i < nCount; i++)
		{
			oYMatrix(i, 0) = oY[i];
		}

		// create the X matrix
		for (size_t nRow = 0; nRow < nCount; nRow++)
		{
			double nVal = 1.0;
			for (int nCol = 0; nCol < nDegree; nCol++)
			{
				oXMatrix(nRow, nCol) = nVal;
				nVal *= oX[nRow];
			}
		}

		// transpose X matrix
		matrix<double> oXtMatrix(trans(oXMatrix));
		// multiply transposed X matrix with X matrix
		matrix<double> oXtXMatrix(prec_prod(oXtMatrix, oXMatrix));
		// multiply transposed X matrix with Y matrix
		matrix<double> oXtYMatrix(prec_prod(oXtMatrix, oYMatrix));

		// lu decomposition
		permutation_matrix<int> pert(oXtXMatrix.size1());
		const std::size_t singular = lu_factorize(oXtXMatrix, pert);
		// must be singular
		BOOST_ASSERT(singular == 0);

		// backsubstitution
		lu_substitute(oXtXMatrix, pert, oXtYMatrix);

		// copy the result to coeff
		return std::vector<double>(oXtYMatrix.data().begin(), oXtYMatrix.data().end());

	}
}