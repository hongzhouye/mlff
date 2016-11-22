#include "include/sgd.hpp"

void SGD::_init_ (const VectorXd& params, double lbd,
    const vvVectorXd& Vb, const vVectorXd& Fb,
    const vvVectorXd& Vt, const vVectorXd& Ft)
{
    Nbasis = Vb.size ();    Ntest = Vt.size ();
    alpha = params.head (Nbasis);
    //cout << "alpha: " << alpha.transpose () << endl;
    gamma = params(Nbasis);
    lambda = lbd;
    Vbasis.assign (Vb.begin (), Vb.end ());
    Fbasis.assign (Fb.begin (), Fb.end ());
    Vtest.assign (Vt.begin (), Vt.end ());
    Ftest.assign (Ft.begin (), Ft.end ());

    g_alpha.setZero (Nbasis);
}

void SGD::_SGD_ ()
{
    int iter = 0, sample_per_batch = Ntest / Nbatch;
    double learning_rate;
    while (iter < MAXITER)
    {
        for (int I = 0; I < Nbatch; I++)
        {
            iter ++;
            learning_rate = pow (tau0 + iter, -kappa);

            batch_start = I * sample_per_batch;
            batch_end = batch_start + sample_per_batch;

            VectorXd g_alp; g_alp.setZero (alpha.rows ());
            double g_gam = 0.;
            for (int i = batch_start; i < batch_end; i++)
            {
                _grad_ (Vtest[i], Ftest[i]);
                g_alp += g_alpha;   g_gam += g_gamma;
            }

            alpha -= learning_rate * (g_alp + lambda * alpha);
            gamma -= learning_rate * g_gam;
        }
        printf ("%6d\t%9.6f\n", iter, _MAE_ ());
    }
}

double SGD::_MAE_ ()
{
    double MAE = 0.;
    MatrixXd G;
    for (int i = 0; i < Ntest; i++)
    {
        _form_G_ (Vtest[i], G);
        MAE += (G * alpha - Ftest[i]).array ().abs ().sum ()
            / (double) Ftest[i].rows ();
    }
    return MAE / (double) Ntest;
}

double SGD::_loss_ ()
{
    double loss = 0.;
    MatrixXd G;
    for (int i = 0; i < Ntest; i++)
    {
        _form_G_ (Vtest[i], G);
        loss += (G * alpha - Ftest[i]).squaredNorm ();
    }
    return 0.5 * loss / (double) Ntest;
}

inline double SGD::_penalized_loss_ ()
{
    return _loss_ () + 0.5 * lambda * alpha.squaredNorm ();
}

void SGD::_form_G_ (const vVectorXd& Vt, MatrixXd& G)
{
    G.setZero (3, Nbasis);
    for (int mu = 0; mu < 3; mu++)
        for (int j = 0; j < Nbasis; j++)
        {
            double dVsq = (Vt[mu] - Vbasis[j][mu]).squaredNorm ();
            G(mu, j) = Fbasis[j][mu] * exp (- gamma * dVsq);
        }
}

void SGD::_form_GH_ (const vVectorXd& Vt, MatrixXd& G, MatrixXd& H)
{
    G.setZero (3, Nbasis);  H = G;
    for (int mu = 0; mu < 3; mu++)
        for (int j = 0; j < Nbasis; j++)
        {
            double dVsq = (Vt[mu] - Vbasis[j][mu]).squaredNorm ();
            G(mu, j) = Fbasis[j][mu] * exp (- gamma * dVsq);
            H(mu, j) = - G(mu, j) * dVsq;
        }
}

void SGD::_grad_ (const vVectorXd& Vt, const VectorXd Ft)
{
    MatrixXd G, H;  _form_GH_ (Vt, G, H);
    VectorXd pred_F = G * alpha;
    g_alpha = G.transpose () * (pred_F - Ft) + lambda * alpha;
    g_gamma = (pred_F - Ft).transpose () * H * alpha;
}
