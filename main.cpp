#include "include/utils.hpp"
#include "include/lattice.hpp"
#include "include/mlfftrain.hpp"
#include "include/sgd.hpp"
#include "include/read.hpp"

int main (int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: inp" << endl;
        exit (1);
    }

    string fname_inp (argv[1]);
    LATTICE lat;
    MLFFTRAIN fft;
    SGD sgd;
    _read_inp_ (fname_inp, lat, fft, sgd);
    lat._fingerprint_ ();
    lat._write_VF_ ();
    lat._shuffle_fingerprint_ ();
    fft._init_ ();
    fft._train_ (lat, sgd);

    return 0;
}
