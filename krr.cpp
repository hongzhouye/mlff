#include "include/krr.hpp"

void KRR::_init_ (double lbd, double gm)
{
    lambda = lbd;   gamma = gm;
    Ntrain = Vtrain.size ();
    Nvalid = Vvalid.size ();
    //M = Ftrain[0].rows ();
    force_limit = 1E3;
}

void KRR::_clear_all_ ()
{
    Vtrain.clear ();    Ftrain.clear ();
    Vvalid.clear ();    Fvalid.clear ();
}

inline double KRR::_predict_F_ (const VectorXd& Vt, bool flag)
{
    return _form_kernel_ (Vtrain, Vt).dot (alpha);

//        if (flag /*&& Vt[0].norm () + Vt[1].norm () + Vt[2].norm () > 1E-10*/)
/*        {
            cout << "Voriginal:\n" << Vt[0].transpose () << endl <<
                Vt[1].transpose () << endl << Vt[2].transpose () << endl << endl;
            cout << "V:\b" << Vtwrap_new[0] << endl << endl;
            cout << "A:\n" << At << endl << endl;
            cout << "X:\n" << Xt << endl << endl;
            cout << "F_xformed:\n" << F_xformed.transpose () << endl << endl;
            cout << "Ft:\n" << Ft.transpose () << endl << endl;
        }*/
}

VectorXd KRR::_predict_F_ (const vVectorXd& Vt)
{
    M = Vt.size ();
    VectorXd pred_F(M);
    for (int mu = 0; mu < M; mu++)
        pred_F(mu) = _predict_F_ (Vt[mu]);
    return pred_F;
}

double KRR::_MAE_ (const vVectorXd& V, dv1& F)
{
    Map<VectorXd> Fp(F.data (), F.size ());
    return _MAE_ (V, Fp);
}

double KRR::_MAE_ (const vVectorXd& V, const VectorXd& F)
{
    int N = V.size ();
    double MAE = 0.;
    for (int i = 0; i < N; i++)
    {
        double pred_F = _predict_F_ (V[i]);
        double error = fabs (pred_F - F(i));
        MAE += error;
        /*cout << "error: " << error << endl;
        if (error > 1E4)
        {
            cout << "pred_F = " << pred_F.transpose () << endl;
            cout << "F      = " << F[i].transpose () << endl;
            cout << "V[i][x]= " << V[i][0].transpose () << endl;
            cout << "V[i][y]= " << V[i][1].transpose () << endl;
            cout << "V[i][z]= " << V[i][2].transpose () << endl << endl;
        }*/
    }
    return MAE / (double) N;
}

double KRR::_MAX_ (const vVectorXd& V, dv1& F)
{
    Map<VectorXd> Fp(F.data (), F.size ());
    return _MAX_ (V, Fp);
}

double KRR::_MAX_ (const vVectorXd& V, const VectorXd& F)
{
    int N = V.size ();
    double MAX = 0.;
    for (int i = 0; i < N; i++)
    {
        double pred_F = _predict_F_ (V[i]);
        double max = fabs (pred_F - F(i));
        if (max > MAX)  MAX = max;
    }
    return MAX;
}

double KRR::_MAE_ (const VectorXd& V, const double F)
{
    vVectorXd Vwrap;   Vwrap.push_back (V);
    VectorXd Fwrap (1);    Fwrap(0) = F;
    return _MAE_ (Vwrap, Fwrap);
}

/*double KRR::_MARE_ (const vvVectorXd& V, const vVectorXd& F)
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
}*/

inline double KRR::_kernel_ (const VectorXd& v1, const VectorXd& v2, double gamma)
{
    return exp (- gamma * (v1 - v2).squaredNorm ());
}

MatrixXd KRR::_form_kernel_ (const vVectorXd& Vt, const vVectorXd& Vtp)
{
    int Nt = Vt.size (), Ntp = Vtp.size ();

    MatrixXd Kt(Nt, Ntp);
    int i, j;
    for (i = 0; i < Nt; i++)    for (j = 0; j < Ntp; j++)
        Kt(i, j) = _kernel_ (Vt[i], Vtp[j], gamma);

    return Kt;
}

VectorXd KRR::_form_kernel_ (const vVectorXd& Vt, const VectorXd& Vtest)
{
    int Nt = Vt.size ();

    int i;
    VectorXd ktest(Nt);
    for (i = 0; i < Nt; i++)    ktest(i) = _kernel_ (Vt[i], Vtest, gamma);

    return ktest;
}

void KRR::_solve_ (string solver)
//  HQ is much faster than CPHQ.
//  Though HQ is said to be less accurate than CPHQ,
//  as far as I can test, there is no difference.
{
    MatrixXd Kt = _form_kernel_ (Vtrain, Vtrain);
    Map<VectorXd> Ft(Ftrain.data (), Ftrain.size ());

    if (solver == "CPHQ")
        alpha = (/*Kt[mu] * */(Kt + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain))).
            colPivHouseholderQr ().solve (/*Kt[mu] * */Ft);
    else if (solver == "HQ")
        alpha = (/*Kt[mu] * */(Kt + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain))).
            householderQr ().solve (/*Kt[mu] * */Ft);

    //printf ("lbd = %5.3e\ttrain MAE = %9.6f\tvalid MAE = %9.6f\t|alpha| = %9.6f\n",
    //    lambda, _MAE_ (Vtrain, Ftrain), _MAE_ (Vvalid, Fvalid),
    //    alpha.norm () / Ntrain);
}

void KRR::_cmp_forces_ (const vVectorXd& V, dv1& Fp, const string& fname)
{
    int N = V.size ();

    Map<VectorXd> F(Fp.data (), Fp.size ());

    bool flag;
    FILE *p = fopen (fname.c_str (), "w+");
    for (int i = 0; i < N; i++)
    {
        double pred_F = _predict_F_ (V[i]);
        fprintf (p, "%9.6f\t%9.6f\n", F(i), pred_F);
    }
    fclose (p);
}

vVectorXd KRR::_comput_forces_ (const vvVectorXd& V)
{
    int N = V.size ();

    vVectorXd Fapp;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        for (int mu = 0; mu < pred_F.rows (); mu++)
            if (fabs (pred_F[mu]) > 1E2)    pred_F(mu) = 0.;
        Fapp.push_back (pred_F);
    }

    return Fapp;
}
