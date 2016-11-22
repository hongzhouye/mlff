#ifndef _SGD_HPP_
#define _SGD_HPP_

#include "utils.hpp"

class SGD
{
    private:
        int batch_start, batch_end;
        void _form_G_ (const vVectorXd&, MatrixXd&);
        void _form_GH_ (const vVectorXd&, MatrixXd&, MatrixXd&);
    public:
//  read from input file
        double tau0, kappa;
        int MAXITER, Nbatch;

//  from MLFFTRAIN
        VectorXd alpha, g_alpha;
        double gamma, g_gamma;
        double lambda;
        vVectorXd Fbasis, Ftest;
        vvVectorXd Vbasis, Vtest;
        int Nbasis, Ntest;

//  member function
        void _init_ (const VectorXd&, double,
            const vvVectorXd&, const vVectorXd&,
            const vvVectorXd&, const vVectorXd&);
        void _SGD_ ();
        double _loss_ ();
        double _MAE_ ();
        inline double _penalized_loss_ ();
        void _grad_ (const vVectorXd&, const VectorXd);
};

#endif
