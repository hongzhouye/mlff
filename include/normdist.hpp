#ifndef _NORMDIST_HPP_
#define _NORMDIST_HPP_

#include <random>
#include "utils.hpp"

void _norm_gen_ (unsigned seed, int Ntot, dv1& pool);
void _norm_gen_ (unsigned seed, int Ntot, VectorXd& pool, double mean, double sigma);

#endif
