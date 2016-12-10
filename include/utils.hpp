#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <chrono>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

typedef vector<string> vs;
typedef vector<int> iv1;
typedef vector<iv1> iv2;
typedef vector<double> dv1;
typedef vector<dv1> dv2;
typedef vector<dv2> dv3;
typedef vector<MatrixXd> vMatrixXd;
typedef vector<VectorXd> vVectorXd;
typedef vector<vVectorXd> vvVectorXd;

void _log_space_ (dv1&, int, double, double);
void _fancy_print_ (string, int space = 6);
void _progress_bar_ (double progress);

#endif
