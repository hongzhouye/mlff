#include "include/mlfftrain.hpp"

#define UP(i,N) (i>=N)?(i-N):(i)
#define DOWN(i,N) (i<0)?(i+N):(i)
void _k_fold_partition_ (const VectorXd& v, VectorXd& vp, int k, int K)
{
    int N = v.rows (), Np = N / K * (K - 1), Nr = N - Np, istart = N / K * k;
    vp.setZero (Np);    //vr.setZero (N - Np);
    for (int i = 0; i < Np; i++)    vp(i) = v(UP(i+istart,N));
}

template <typename T>
void _k_fold_partition_ (const vector<T>& v, vector<T>& vp, vector<T>& vr,
    int k, int K)
{
    int N = v.size (), Np = N / K * (K - 1), Nr = N - Np, istart = N / K * k;
    for (int i = 0; i < Np; i++)    vp.push_back (v[UP(i+istart,N)]);
    for (int i = 0; i < Nr; i++)    vr.push_back (v[DOWN(istart-i-1,N)]);
}
#undef UP
#undef DOWN

void MLFFTRAIN::_train_ (const LATTICE& lat)
{
    vvVectorXd Vtrain_all;
    Vtrain_all.assign (lat.V.begin (), lat.V.begin () + Ntrain);
    vVectorXd Ftrain_all;
    Ftrain_all.assign (lat.F.begin (), lat.F.begin () + Ntrain);

    printf ("lambda\tvalid MAE\n");
    for (auto i = lbd_set.begin (); i < lbd_set.end (); i++)
    {
        double lbd = *i;
        double valid_MAE = 0.;
        for (int k = 0; k < K; k++)
        {
            krr._clear_all_ ();
            _k_fold_partition_ (Vtrain_all, krr.Vtrain, krr.Vvalid, k, K);
            _k_fold_partition_ (Ftrain_all, krr.Ftrain, krr.Fvalid, k, K);

            krr._init_ (lbd, gamma);
            krr._solve_ ();
            valid_MAE += krr._MAE_ (krr.Vvalid, krr.Fvalid);
        }
        printf ("%5.3e\t%9.6f\n", lbd, valid_MAE / K);
    }
}

void MLFFTRAIN::_write_VF_ ()
{
    if (V.size () != F.size ())
    {
        cout << "[Error] In MLFFTRAIN: V and F must have the same size." << endl;
        exit (1);
    }

    FILE *px = fopen ("output/training_VFx.dat", "w+");
    FILE *py = fopen ("output/training_VFy.dat", "w+");
    FILE *pz = fopen ("output/training_VFz.dat", "w+");
    for (int i = 0; i < V.size (); i++)
    {
        for (int j = 0; j < V[i][0].rows (); j++)
        {
            fprintf (px, "%9.6f;", V[i][0][j]);
            fprintf (py, "%9.6f;", V[i][1][j]);
            fprintf (pz, "%9.6f;", V[i][2][j]);
        }
        fprintf (px, "%9.6f\n", F[i][0]);
        fprintf (py, "%9.6f\n", F[i][1]);
        fprintf (pz, "%9.6f\n", F[i][2]);
    }
    fclose (px);    fclose (py);    fclose (pz);
}
