#ifndef _MLFFTRAIN_HPP_
#define _MLFFTRAIN_HPP_

#include "utils.hpp"
#include "normdist.hpp"
#include "lattice.hpp"
#include "sgd.hpp"

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
    public:
//  from input
        int K, Neta, Ntrain, Nlbd;
        dv1 eta, lbd_set;
        double Rc;

//  param's
        double gamma;
        VectorXd alpha;
        vVectorXd F;
        vvVectorXd V;

//  member function
        void _init_ ();
        void _train_ (const LATTICE&, SGD&);
        void _write_VF_ ();
};

#endif
