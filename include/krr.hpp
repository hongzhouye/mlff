#ifndef _KRR_HPP_
#define _KRR_HPP_

#include "utils.hpp"

class KRR
{
    private:
        int batch_start, batch_end;
        void _form_G_ (const vVectorXd&, MatrixXd&);
        void _form_GH_ (const vVectorXd&, MatrixXd&, MatrixXd&);

    public:
//  from MLFFTRAIN
        VectorXd alpha;
        double gamma;
        double lambda;
        dv1 Fbasis, Ftrain, Fvalid;
        vVectorXd Vbasis, Vtrain, Vvalid;
        int Nbasis, Ntrain, Nvalid, M;
        vMatrixXd Atrain, Xtrain;
        double force_limit;

//  member function
        void _init_ (double, double);
        void _clear_all_ ();
        inline double _predict_F_ (const VectorXd& Vt, bool flag = false);
        VectorXd _predict_F_ (const vVectorXd& Vt);
        double _loss_ (const vvVectorXd& V, const vVectorXd& F);
        double _MAE_ (const vVectorXd& V, dv1& F);
        double _MAE_ (const vVectorXd& V, const VectorXd& F);
        double _MAX_ (const vVectorXd& V, dv1& F);
        double _MAX_ (const vVectorXd& V, const VectorXd& F);
        double _MAE_ (const VectorXd& V, const double F);
        double _MARE_ (const vvVectorXd& V, const vVectorXd& F);
        inline double _penalized_loss_ (const vvVectorXd& V, const vVectorXd& F);

        inline double _kernel_ (const VectorXd& v1, const VectorXd& v2, double gamma);
        vMatrixXd _fingerprint_xform_ (const vvVectorXd& Vt);
        vMatrixXd _V_to_A_ (const vMatrixXd& Vt);
        vMatrixXd _form_X_ (const vMatrixXd& Vt, const vMatrixXd& At);
        MatrixXd _form_kernel_ (const vVectorXd& Vt, const vVectorXd& Vtp);
        VectorXd _form_kernel_ (const vVectorXd& Vt, const VectorXd& Vtest);
        MatrixXd _form_force_mat_ (const vVectorXd& Ft, const vMatrixXd& At);
        void _solve_ (string);
        void _cmp_forces_ (const vVectorXd& V, dv1& Fp, const string&);
        vVectorXd _comput_forces_ (const vvVectorXd& V);
};

#endif
