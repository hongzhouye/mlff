#include "include/krr.hpp"

void KRR::_init_ (double lbd, double gm)
{
    lambda = lbd;   gamma = gm;
    Ntrain = Vtrain.size ();
    Nvalid = Vvalid.size ();
    M = Ftrain[0].rows ();
    force_limit = 1E3;
}

void KRR::_clear_all_ ()
{
    Vtrain.clear ();    Ftrain.clear ();
    Vvalid.clear ();    Fvalid.clear ();
}

inline VectorXd KRR::_predict_F_ (const vVectorXd& Vt, bool flag)
{
    VectorXd Ft (M);

    double normVt = 0.;
    for (int mu = 0; mu < Vt.size (); mu++) normVt += Vt[mu].norm ();
    if (normVt < 1E-16)
        Ft.setZero ();
    else
    {
        for (int mu = 0; mu < M; mu++)
        {
            vVectorXd kt = _form_kernel_ (Vtrain, Vt);
            Ft(mu) = alpha[mu].dot (kt[mu]);
        }

        for (int mu = 0; mu < M; mu++)
            if (fabs (Ft[mu]) > force_limit) Ft[mu] = 0.;

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

    return Ft;
}

double KRR::_MAE_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size (), M = F[0].rows ();
    double MAE = 0.;
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        double error = (pred_F - F[i]).norm ();
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

double KRR::_MAE_ (const vVectorXd& V, const VectorXd& F)
{
    vvVectorXd Vwrap;   Vwrap.push_back (V);
    vVectorXd Fwrap;    Fwrap.push_back (F);
    return _MAE_ (Vwrap, Fwrap);
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

/*inline double KRR::_penalized_loss_ (const vvVectorXd& V, const vVectorXd& F)
{
    return _loss_ (V, F) + lambda * alpha.squaredNorm ();
}*/

inline double KRR::_kernel_ (const VectorXd& v1, const VectorXd& v2, double gamma)
{
    return exp (- gamma * (v1 - v2).squaredNorm ());
}

vMatrixXd KRR::_form_kernel_ (const vvVectorXd& Vt, const vvVectorXd& Vtp)
{
    int Nt = Vt.size (), Ntp = Vtp.size (), M = Vt[0].size ();

    vMatrixXd Kt_set;
    for (int mu = 0; mu < M; mu++)
    {
        MatrixXd Kt(Nt, Ntp);
        int i, j;
        for (i = 0; i < Nt; i++)    for (j = 0; j < Ntp; j++)
            Kt(i, j) = _kernel_ (Vt[i][mu], Vtp[j][mu], gamma);
        Kt_set.push_back (Kt);
    }
    return Kt_set;
}

vVectorXd KRR::_form_kernel_ (const vvVectorXd& Vt, const vVectorXd& Vtest)
{
    int Nt = Vt.size ();

    vVectorXd ktest;
    int i;
    for (int mu = 0; mu < M; mu++)
    {
        VectorXd k(Nt);
        for (i = 0; i < Nt; i++)
            k(i) = _kernel_ (Vt[i][mu], Vtest[mu], gamma);
        ktest.push_back (k);
    }
    return ktest;
}

vVectorXd _form_Ft_ (const vVectorXd& Ftrain)
{
    int N = Ftrain.size (), M = Ftrain[0].rows ();
    vVectorXd Ft;
    for (int mu = 0; mu < M; mu++)
    {
        VectorXd Fmu (N);
        for (int i = 0; i < N; i++) Fmu(i) = Ftrain[i](mu);
        Ft.push_back (Fmu);
    }
    return Ft;
}

void KRR::_solve_ (string solver)
//  HQ is much faster than CPHQ.
//  Though HQ is said to be less accurate than CPHQ,
//  as far as I can test, there is no difference.
{
    vMatrixXd Kt = _form_kernel_ (Vtrain, Vtrain);
    vVectorXd Ft = _form_Ft_ (Ftrain);

    alpha.clear ();
    for (int mu = 0; mu < M; mu++)
    {
        VectorXd a;
        if (solver == "CPHQ")
            a = (/*Kt[mu] * */(Kt[mu] + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain))).
                colPivHouseholderQr ().solve (/*Kt[mu] * */Ft[mu]);
        else if (solver == "HQ")
            a = (/*Kt[mu] * */(Kt[mu] + lambda * Ntrain * MatrixXd::Identity (Ntrain, Ntrain))).
                householderQr ().solve (/*Kt[mu] * */Ft[mu]);
        //cout << "alpha[" << mu << "]:\n" << a.transpose () << endl;
        alpha.push_back (a);
    }

    //printf ("lbd = %5.3e\ttrain MAE = %9.6f\tvalid MAE = %9.6f\t|alpha| = %9.6f\n",
    //    lambda, _MAE_ (Vtrain, Ftrain), _MAE_ (Vvalid, Fvalid),
    //    alpha.norm () / Ntrain);
}

void KRR::_cmp_forces_ (const vvVectorXd& V, const vVectorXd& F)
{
    int N = V.size (), M = F[0].rows ();

    bool flag;
    FILE *p = fopen ("1.dat", "w+");
    for (int i = 0; i < N; i++)
    {
        VectorXd pred_F = _predict_F_ (V[i]);
        for (int mu = 0; mu < M; mu++)
            fprintf (p, "%9.6f\t%9.6f\n", F[i](mu), pred_F (mu));
    }
    fclose (p);
}

vVectorXd KRR::_comput_forces_ (const vvVectorXd& V)
{
    int N = V.size ();

    vVectorXd Fapp;
    bool flag;
    for (int i = 0; i < N; i++)
    {
        if (i == 27 /*|| i == 28*/)  flag = true;
        else    flag = false;
        VectorXd pred_F = _predict_F_ (V[i], flag);
        for (int mu = 0; mu < pred_F.rows (); mu++)
            if (fabs (pred_F[mu]) > 1E2)    pred_F[mu] = 0.;
        Fapp.push_back (pred_F);
    }

    return Fapp;
}
