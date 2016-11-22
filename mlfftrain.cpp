#include "include/mlfftrain.hpp"

void MLFFTRAIN::_init_ ()
{
//  random initialization for alpha
    alpha.setZero (Ntrain);
    unsigned seed;
    seed = chrono::system_clock::now ().time_since_epoch ().count ();
    _norm_gen_ (seed, Ntrain, alpha, 0., 1. / sqrt ((double) Ntrain));

//  set gamma to zero
    gamma = 0.1;
}

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

void MLFFTRAIN::_train_ (const LATTICE& lat, SGD& sgd)
{
    vvVectorXd Vtrain;  Vtrain.assign (lat.V.begin (), lat.V.begin () + Ntrain);
    vVectorXd Ftrain;   Ftrain.assign (lat.F.begin (), lat.F.begin () + Ntrain);

    for (auto i = lbd_set.begin (); i < lbd_set.end (); i++)
    {
        double lbd = *i;
        double loss = 0.;
        for (int k = 0; k < K; k++)
        {
            int Np = Ntrain / K * (K - 1);
            VectorXd params;
            _k_fold_partition_ (alpha, params, k, K);
            params.conservativeResize (Np + 1);
            params(Np) = gamma;

            vvVectorXd Vbasis, Vtest;
            _k_fold_partition_ (Vtrain, Vbasis, Vtest, k, K);
            vVectorXd Fbasis, Ftest;
            _k_fold_partition_ (Ftrain, Fbasis, Ftest, k, K);

            sgd._init_ (params, lbd, Vbasis, Fbasis, Vtest, Ftest);
            sgd._SGD_ ();
            loss += sgd._loss_ ();
        }
        printf ("lbd = %5.3e\tloss = %9.6f\n", lbd, loss / K);
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
