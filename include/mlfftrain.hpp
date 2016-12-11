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
            vVectorXd& Vtrain, dv1& Ftrain,
            vVectorXd& Vtest, dv1& Ftest,
            LATTICE& lat);
    public:
//  from input
        int K, Neta, Ntrain, Ntest, Nlbd;
        dv1 eta, lbd_set;
        double Rc;

        double Fc;  // for filter train
        string cmp_force;

//  param's
        double gamma;
        VectorXd alpha;
        vVectorXd F;
        vvVectorXd V;

//  xyz independent
        dv1 Findpt;
        vVectorXd Vindpt;

//  Kernel ridge regression class
        KRR krr;

//  member function
        void _krr_basis_ (const LATTICE&);
        void _train_ (LATTICE&);
        void _1by1_train_ (LATTICE&);
        void _app_ (const LATTICE&);
        void _write_VF_ (const vvVectorXd&, const vVectorXd&);
};

#endif
