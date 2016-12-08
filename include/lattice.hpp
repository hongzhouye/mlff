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
        string out_path, out_path_app;
        bool shuf, write;

//  read from data file
        vvVectorXd R, Rapp, Rsanity;
        vVectorXd F, Fsanity;

//  fingerprint
        vvVectorXd V, Vapp, Vsanity;

//  member function
        void _print_ ();
        void _fingerprint_ (const vvVectorXd& R, vvVectorXd& V, vVectorXd& F);
        void _fingerprint_ (const vvVectorXd& R, vvVectorXd& V);
        void _shuffle_fingerprint_ (vvVectorXd& V, vVectorXd& F);
        void _write_VF_ ();

        dv1 _rdf_per_cell_ (const vVectorXd& R, double Rc, const dv1& bin);
        void _gen_rdf_ (int nbin);

        void _count_zero_ (const vvVectorXd& V);
};

#endif
