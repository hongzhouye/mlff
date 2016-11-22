#include "include/normdist.hpp"

void _norm_gen_ (unsigned seed, int Ntot, dv1& pool)
{
	// construct a trivial random generator engine from a time-based seed:
	default_random_engine generator (seed);
	normal_distribution<double> distribution (0., 1.);
	for (int i = 0; i < Ntot; i++)	pool.push_back (distribution(generator));
}

void _norm_gen_ (unsigned seed, int Ntot, VectorXd& pool, double mean, double sigma)
{
	// construct a trivial random generator engine from a time-based seed:
	default_random_engine generator (seed);
	normal_distribution<double> distribution (mean, sigma);
	for (int i = 0; i < Ntot; i++)	pool(i) = distribution(generator);
}
