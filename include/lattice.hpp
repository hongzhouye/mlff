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

        iv2 _form_index_ (const vvVectorXd& V);
    public:
//  read from input file
        dv1 Rp;
        dv1 eta;
        dv1 lat_len, lat_ang;
        double Rc;
        string out_path, out_path_app;
        bool shuf, write;
        double Fmax;

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
        //template <typename T1, typename T2>
        //void _shuffle_fingerprint_ (vector<T1>& V, vector<T2>& F);
        void _shuffle_fingerprint_ (vVectorXd& V, dv1& F);
        void _write_VF_ ();

        dv1 _rdf_per_cell_ (const vVectorXd& R, double Rc, const dv1& bin);
        void _gen_rdf_ (int nbin);

        void _form_sanity_set_ (
            vvVectorXd& Vtrain, vVectorXd& Ftrain,
            vvVectorXd& Vtest, vVectorXd& Ftest, int Nsanity = 10);
};

#endif
