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
    return _form_kernel_ (Vtrain, Vt) * alpha;
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

MatrixXd KRR::_form_kernel_ (const vvVectorXd& Vt)
{
    int Nt = Vt.size ();

    MatrixXd Kt (Nt * M, Nt * M);
    MatrixXd K; K.setZero (Nt, Nt);
    int mu, nu, i, j;
    for (mu = 0; mu < M; mu++) for (nu = 0; nu <= mu; nu++)
    {
        if (mu == nu)
            for (i = 0; i < Nt; i++)    for (j = 0; j <= i; j++)
                K(i, j) = K(j, i) = _kernel_ (Vt[i][mu], Vt[j][nu], gamma);
        else
            for (i = 0; i < Nt; i++)    for (j = 0; j < Nt; j++)
                K(i, j) = _kernel_ (Vt[i][mu], Vt[j][nu], gamma);

        Kt.block (mu * Nt, nu * Nt, Nt, Nt) = K;
        if (mu != nu)   Kt.block (nu * Nt, mu * Nt, Nt, Nt) = K.transpose ();
    }

    return Kt;
}

MatrixXd KRR::_form_kernel_ (const vvVectorXd& Vt, const vVectorXd& Vtest)
{
    int Nt = Vt.size ();

    MatrixXd ktest (Nt * M, M);
    VectorXd k; k.setZero (Nt);
    int mu, nu, i;
    for (mu = 0; mu < M; mu++) for (nu = 0; nu < M; nu++)
    {
        for (i = 0; i < Nt; i++)
            k(i) = _kernel_ (Vtest[mu], Vt[i][nu], gamma);
        ktest.block (mu * Nt, nu, Nt, 1) = k;
    }
    ktest.transposeInPlace ();
    return ktest;
}

VectorXd KRR::_form_force_vec_ (const vVectorXd& Ft)
{
    int Nt = Ft.size ();

    VectorXd F(M * Nt), f(Nt);
    int mu, i;
    for (mu = 0; mu < M; mu++)
    {
        for (i = 0; i < Nt; i++)    f(i) = Ft[i](mu);
        F.block (mu * Nt, 0, Nt, 1) = f;
    }
    return F;
}

void KRR::_solve_ ()
{
    MatrixXd Kt = _form_kernel_ (Vtrain);
    //cout << "Kt:\n" << Kt << endl << endl;
    VectorXd Ft = _form_force_vec_ (Ftrain);
    //cout << "Ft:\n" << Ft << endl << endl;

    alpha = (Kt + lambda * Ntrain *
        MatrixXd::Identity (Ntrain * M, Ntrain * M)).
        colPivHouseholderQr ().solve (Ft);

    printf ("lbd = %5.3e\ttrain MAE = %9.6f\tvalid MAE = %9.6f\n",
        lambda, _MAE_ (Vtrain, Ftrain), _MAE_ (Vvalid, Fvalid));
}
