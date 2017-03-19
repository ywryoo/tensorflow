#include <vector>
#include <iostream>
#include <random>
#include "armadillo"

using namespace arma;
using namespace std;

int main(void) {
mat  A  = randu<mat>(5,5);
mat  B  = randu<mat>(5,5);

cout << A << endl;

uvec q1 = find(A > B);
uvec q2 = find(A > 0.5);
uvec q3 = find(A > 0.5, 3, "last");

cout << q2 << endl;

// change elements of A greater than 0.5 to 1
A.elem( find(A > 0.5) ).ones();	

	return 0;
}
