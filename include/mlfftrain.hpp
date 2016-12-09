#ifndef _MLFFTRAIN_HPP_
#define _MLFFTRAIN_HPP_

#include "utils.hpp"
#include "lattice.hpp"
#include "krr.hpp"

/*************************************************************/
/*       CLASS: Machine Learning Force Field - Training      */
/*************************************************************/
//  K           fold of cross-validation
//  Neta        # of eta's
//  etamin/max  range of eta grid
//  Ntrain      training set size
//  Nlbd        # of lambda's
//  lbdmin/max  range of lambda (to be searched)
/*************************************************************/

class MLFFTRAIN
{
    private:
        void _form_training_test_set_ (
            vvVectorXd& Vtrain, vVectorXd& Ftrain,
            vvVectorXd& Vtest, vVectorXd& Ftest,
            LATTICE& lat);
    public:
//  from input
        int K, Neta, Ntrain, Ntest, Nlbd;
        dv1 eta, lbd_set;
        double Rc;

//  param's
        double gamma;
        VectorXd alpha;
        vVectorXd F;
        vvVectorXd V;

//  Kernel ridge regression class
        KRR krr;

//  member function
        void _krr_basis_ (const LATTICE&);
        void _train_ (LATTICE&);
        void _app_ (const LATTICE&);
        void _write_VF_ ();
};

#endif
