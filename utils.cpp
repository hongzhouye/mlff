#include "include/utils.hpp"

void _log_space_ (dv1& A, int NA, double min, double max)
{
    double delta = (log (max) - log (min)) / (NA - 1.);
    if (A.size () == NA)
        for (int i = 0; i < NA; i++)    A[i] = min * exp (i * delta);
    else
        for (int i = 0; i < NA; i++)    A.push_back (min * exp (i * delta));
}

void _fancy_print_ (string fancy, int space)
{
	int n = fancy.length ();
	int len = (space + 1) * 2 + n;
	int i;
	for (i = 0; i < len; i++)	cout << "=";
	cout << endl << "|";
	for (i = 0; i < space; i++)	cout << " ";
	cout << fancy;
	for (i = 0; i < space; i++)	cout << " ";
	cout << "|\n";
	for (i = 0; i < len; i++)	cout << "=";
	cout << endl;
}
