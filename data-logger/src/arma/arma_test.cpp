#include <armadillo>
#include <iostream>
int main(int argc, char** argv)
{
  arma::mat A(10, 8, arma::fill::zeros);
  A.load("C:\\Users\\ttw2x\\Documents\\git_repos\\deepracing\\data-logger\\tracks\\Australia_racingline.arma.txt", arma::arma_ascii);
  std::cout << A << std::endl;

}