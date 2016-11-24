#include "include/krr.hpp"

void KRR::_init_ (double lbd, double gm)
{
    lambda = lbd;   gamma = gm;
    Nbasis = Vbasis.size ();
    Ntrain = Vtrain.size ();
    Nvalid = Vvalid.size ();
    M = Fbasis[0].rows ();
}

void KRR::_clear_all_ ()
{
    Vtrain.clear ();    Ftrain.clear ();
    Vvalid.clear ();    Fvalid.clear ();
}

VectorXd KRR::_predict_F_ (const vVectorXd& Vt)
{
    vVectorXd Kt = _form_kernel_ (Vt, Vbasis, Fbasis);
    VectorXd pred_F (M);
    for (int mu = 0; mu < M; mu++)  pred_F(mu) = Kt[mu].dot (alpha);
    return pred_F;
}

double KRR::_MAE_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size ();
    double MAE = 0.;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        MAE += (pred_F - F[i]).array ().abs ().sum ()
            / (double) F[i].rows ();
    }
    return MAE / (double) N;
}

double KRR::_loss_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size ();
    double loss = 0.;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        loss += (pred_F - F[i]).squaredNorm ();
    }
    return loss / (double) N;
}

inline double KRR::_penalized_loss_ (const vvVectorXd& V, const vVectorXd& F)
{
    return _loss_ (V, F) + lambda * alpha.squaredNorm ();
}

inline double _kernel_ (const VectorXd& v1, const VectorXd& v2, double gamma)
{
    return exp (- gamma * (v1 - v2).squaredNorm ());
}

vMatrixXd KRR::_form_kernel_ (const vvVectorXd& Vt,
    const vvVectorXd& Vb, const vVectorXd& Fb)
{
    int Nt = Vt.size (), Nb = Vb.size (), M = Fb[0].size ();
    Kt.clear ();
    for (int mu = 0; mu < M; mu++)
    {
        MatrixXd Ktmu;  Ktmu.setZero (Nt, Nb);
        for (int i = 0; i < Nt; i++)
        {
            for (int j = 0; j < Nb; j++)
                Ktmu(i, j) = _kernel_ (Vt[i][mu], Vb[j][mu], gamma)
                    * Fb[j](mu);
        }
        Kt.push_back (Ktmu);
    }
    return Kt;
}

vVectorXd KRR::_form_kernel_ (const vVectorXd& Vt,
    const vvVectorXd& Vb, const vVectorXd& Fb)
{
    vvVectorXd Vtp; Vtp.push_back (Vt);
    vMatrixXd Kt = _form_kernel_ (Vtp, Vb, Fb);
    vVectorXd Ktp;
    for (int mu = 0; mu < Kt.size (); mu++)
        Ktp.push_back (Kt[mu].row (0));
    return Ktp;
}

void KRR::_solve_ ()
{
    vMatrixXd Kt = _form_kernel_ (Vtrain, Vbasis, Fbasis);
    MatrixXd A; A.setZero (Nbasis, Nbasis);
    VectorXd b; b.setZero (Nbasis);
    for (int mu = 0; mu < M; mu++)
    {
        A += Kt[mu].transpose () * Kt[mu];
        VectorXd Ft;    Ft.setZero (Ntrain);
        for (int i = 0; i < Ntrain; i++) Ft(i) = Ftrain[i](mu);
        b += Kt[mu].transpose () * Ft;
    }
    A += lambda * Ntrain * MatrixXd::Identity (Nbasis, Nbasis);
    alpha = A.colPivHouseholderQr().solve (b);

    printf ("lbd = %5.3e\ttrain MAE = %9.6f\tvalid MAE = %9.6f\n",
        lambda, _MAE_ (Vtrain, Ftrain), _MAE_ (Vvalid, Fvalid));
}
