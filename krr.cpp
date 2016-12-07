#include "include/krr.hpp"

void KRR::_init_ (double lbd, double gm)
{
    lambda = lbd;   gamma = gm;
    Ntrain = Vtrain.size ();
    Nvalid = Vvalid.size ();
    M = Ftrain[0].rows ();
}

void KRR::_clear_all_ ()
{
    Vtrain.clear ();    Ftrain.clear ();
    Vvalid.clear ();    Fvalid.clear ();
}

inline VectorXd KRR::_predict_F_ (const vVectorXd& Vt)
{
    vvVectorXd Vtwrap;  Vtwrap.push_back (Vt);
    vMatrixXd Vtwrap_new = _fingerprint_xform_ (Vtwrap);
    vMatrixXd Atwrap = _V_to_A_ (Vtwrap_new);
    vMatrixXd Xtwrap = _form_X_ (Vtwrap_new, Atwrap);
    MatrixXd At = Atwrap[0], Xt = Xtwrap[0];

    VectorXd F_xformed = alpha.transpose () * _form_kernel_ (Xtrain, Xt);
    VectorXd Ft = At.colPivHouseholderQr ().solve (F_xformed);

    return Ft;
}

double KRR::_MAE_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size (), M = F[0].rows ();
    double MAE = 0.;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        MAE += (pred_F - F[i]).array ().abs ().sum ()
            / (double) M;
    }
    return MAE / (double) N;
}

double KRR::_MARE_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size (), M = F[0].rows ();
    double MARE = 0.;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        MARE += ((pred_F - F[i]).array () / F[i].array ()).abs ().sum ()
            / (double) M;
    }
    return MARE / (double) N;
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

inline double KRR::_kernel_ (const MatrixXd& v1, const MatrixXd& v2, double gamma)
{
    return exp (- gamma * (v1 - v2).squaredNorm ());
}

MatrixXd KRR::_form_kernel_ (const vMatrixXd& Vt, const vMatrixXd& Vtp)
{
    int Nt = Vt.size (), Ntp = Vtp.size ();

    MatrixXd Kt(Nt, Ntp);
    int i, j;
    for (i = 0; i < Nt; i++)    for (j = 0; j < Ntp; j++)
        Kt(i, j) = _kernel_ (Vt[i], Vtp[j], gamma);
    return Kt;
}

VectorXd KRR::_form_kernel_ (const vMatrixXd& Vt, const MatrixXd& Vtest)
{
    int Nt = Vt.size ();

    VectorXd ktest(Nt);
    int i;
    for (i = 0; i < Nt; i++)    ktest(i) = _kernel_ (Vt[i], Vtest, gamma);

    return ktest;
}

MatrixXd KRR::_form_force_mat_ (const vVectorXd& Ft, const vMatrixXd& At)
{
    int Nt = Ft.size (), Neta = At[0].rows ();

    MatrixXd Ft_new(Nt, Neta);
    VectorXd F_xformed (Neta);
    for (int i = 0; i < Nt; i++)
    {
        F_xformed = At[i] * Ft[i];
        Ft_new.row (i) = F_xformed.transpose ();
    }
    return Ft_new;
}

vMatrixXd KRR::_fingerprint_xform_ (const vvVectorXd& Vt)
{
    int Nt = Vt.size (), M = Vt[0].size (), Neta = Vt[0][0].rows ();

    vMatrixXd Vt_new;
    MatrixXd V(Neta, M);
    int mu, i, j;
    for (i = 0; i < Nt; i++)
    {
        for (mu = 0; mu < M; mu++)  for (j = 0; j < Neta; j++)
            V(j, mu) = Vt[i][mu][j];
        Vt_new.push_back (V);
    }
    return Vt_new;
}

vMatrixXd KRR::_V_to_A_ (const vMatrixXd& Vt)
{
    int Nt = Vt.size (), Neta = Vt[0].rows (), M = Vt[0].cols ();

    vMatrixXd At;
    MatrixXd Vh(Neta, M);
    for (int i = 0; i < Nt; i++)
    {
        for (int j = 0; j < Neta; j++)
            Vh.row (j) = Vt[i].row (j) / Vt[i].row (j).norm ();
        At.push_back (Vh);
    }
    return At;
}

vMatrixXd KRR::_form_X_ (const vMatrixXd& Vt, const vMatrixXd& At)
{
    int Nt = Vt.size ();

    vMatrixXd Xt;
    for (int i = 0; i < Nt; i++)
        Xt.push_back (Vt[i] * At[i].transpose ());

    return Xt;
}

void KRR::_solve_ (string solver)
//  HQ is much faster than CPHQ.
//  Though HQ is said to be less accurate than CPHQ,
//  as far as I can test, there is no difference.
{
    Atrain.clear ();    Xtrain.clear ();
    vMatrixXd Vtrain_new = _fingerprint_xform_ (Vtrain);
    Atrain = _V_to_A_ (Vtrain_new);
    Xtrain = _form_X_ (Vtrain_new, Atrain);

    /*_fancy_print_ ("check V (PRL)", 2);
    for (int i = 0; i < Ntrain; i++)
        cout << i << ":\n" << Xtrain[i] << endl << endl;*/

    MatrixXd Kt = _form_kernel_ (Xtrain, Xtrain);
    //cout << "Kt:\n" << Kt << endl << endl;
    MatrixXd Ft = _form_force_mat_ (Ftrain, Atrain);
    //cout << "Ft:\n" << Ft << endl << endl;

    if (solver == "CPHQ")
        alpha = (Kt + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain)).
            colPivHouseholderQr ().solve (Ft);
    else if (solver == "HQ")
        alpha = (Kt + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain)).
            householderQr ().solve (Ft);

    //printf ("lbd = %5.3e\ttrain MAE = %9.6f\tvalid MAE = %9.6f\t|alpha| = %9.6f\n",
    //    lambda, _MAE_ (Vtrain, Ftrain), _MAE_ (Vvalid, Fvalid), alpha.norm () / Ntrain);
}
