#ifndef _LATTICE_HPP_
#define _LATTICE_HPP_

#include "utils.hpp"

class LATTICE
{
    private:
        double _periodic_ (dv1&, const dv1&, double);
        void _gen_neighbor_list_ (dv3&, dv2&,
            const vMatrixXd&, const MatrixXd&, double);
        vvVectorXd _R_to_V_ (const vVectorXd&, double, const dv1&);
    public:
//  read from input file
        int Neta;
        dv1 eta;
        dv1 lat_len, lat_ang;
        double Rc;
        string out_path;
        bool shuf, write;

//  read from data file
        vvVectorXd R;
        vVectorXd F;

//  fingerprint
        vvVectorXd V;

//  member function
        void _fingerprint_ ();
        void _shuffle_fingerprint_ ();
        void _write_VF_ ();
};

#endif
