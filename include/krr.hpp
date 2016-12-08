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
        MatrixXd alpha;
        double gamma;
        double lambda;
        vVectorXd Fbasis, Ftrain, Fvalid;
        vvVectorXd Vbasis, Vtrain, Vvalid;
        int Nbasis, Ntrain, Nvalid, M;
        vMatrixXd Atrain, Xtrain;

//  member function
        void _init_ (double, double);
        void _clear_all_ ();
        inline VectorXd _predict_F_ (const vVectorXd&, bool flag = false);
        double _loss_ (const vvVectorXd& V, const vVectorXd& F);
        double _MAE_ (const vvVectorXd&, const vVectorXd&);
        double _MARE_ (const vvVectorXd& V, const vVectorXd& F);
        inline double _penalized_loss_ (const vvVectorXd& V, const vVectorXd& F);

        inline double _kernel_ (const MatrixXd& v1, const MatrixXd& v2, double gamma);
        vMatrixXd _fingerprint_xform_ (const vvVectorXd& Vt);
        vMatrixXd _V_to_A_ (const vMatrixXd& Vt);
        vMatrixXd _form_X_ (const vMatrixXd& Vt, const vMatrixXd& At);
        MatrixXd _form_kernel_ (const vMatrixXd& Vt, const vMatrixXd& Vtp);
        VectorXd _form_kernel_ (const vMatrixXd& Vt, const MatrixXd& Vtest);
        MatrixXd _form_force_mat_ (const vVectorXd& Ft, const vMatrixXd& At);
        void _solve_ (string);
        void _cmp_forces_ (const vvVectorXd& V, const vVectorXd& F);
        vVectorXd _comput_forces_ (const vvVectorXd& V);
};

#endif
