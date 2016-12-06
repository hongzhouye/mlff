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
        vVectorXd Fbasis, Ftrain, Fvalid;
        vvVectorXd Vbasis, Vtrain, Vvalid;
        int Nbasis, Ntrain, Nvalid, M;

//  member function
        void _init_ (double, double);
        void _clear_all_ ();
        inline VectorXd _predict_F_ (const vVectorXd&);
        double _loss_ (const vvVectorXd& V, const vVectorXd& F);
        double _MAE_ (const vvVectorXd&, const vVectorXd&);
        inline double _penalized_loss_ (const vvVectorXd& V, const vVectorXd& F);

        MatrixXd _form_kernel_ (const vvVectorXd& Vt);
        MatrixXd _form_kernel_ (const vvVectorXd& Vt, const vVectorXd& Vtest);
        VectorXd _form_force_vec_ (const vVectorXd& Ft);
        void _solve_ ();
};

#endif
