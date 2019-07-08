#include <armadillo>
#include <iostream>
int main(int argc, char** argv)
{
  arma::mat A(10, 8, arma::fill::zeros);
  std::cout << A << std::endl;
  A.save("identity.arma", arma::arma_ascii);

}