#include "include/utils.hpp"
#include "include/lattice.hpp"
#include "include/mlfftrain.hpp"
#include "include/krr.hpp"
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
    _read_inp_ (fname_inp, lat, fft);
    lat._fingerprint_ (lat.R, lat.V, lat.F);
    //lat._fingerprint_ (lat.Rsanity, lat.Vsanity, lat.Fsanity);
    lat._fingerprint_ (lat.Rapp, lat.Vapp);
    //lat._gen_rdf_ (200);
    //fft._1by1_train_ (lat);
    fft._train_ (lat);
    fft._app_ (lat);

    return 0;
}
