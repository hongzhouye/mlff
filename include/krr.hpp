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
        vMatrixXd Kt;

//  member function
        void _init_ (double, double);
        void _clear_all_ ();
        VectorXd _predict_F_ (const vVectorXd&);
        double _loss_ (const vvVectorXd& V, const vVectorXd& F);
        double _MAE_ (const vvVectorXd&, const vVectorXd&);
        inline double _penalized_loss_ (const vvVectorXd& V, const vVectorXd& F);

        vMatrixXd _form_kernel_ (const vvVectorXd& Vt,
            const vvVectorXd& Vb, const vVectorXd& Fb);
        vVectorXd _form_kernel_ (const const vVectorXd& Vt,
            const vvVectorXd& Vb, const vVectorXd& Fb);
        void _solve_ ();
};

#endif
