#include "include/lattice.hpp"

void _gen_dist_mat_ (const vVectorXd& R, vMatrixXd& dR, MatrixXd& dist)
{
    int N = R.size (), M = R[0].size ();
    dist.setZero (N, N);
    for (int mu = 0; mu < M; mu++)  dR.push_back (dist);

    for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++)
        if (i == j) continue;
        else
        {
            for (int mu = 0; mu < M; mu++)
            {
                double delta = R[j][mu] - R[i][mu];
                dR[mu](i, j) = delta;   dR[mu](j, i) = -dR[mu](i, j);
                dist(i, j) += delta * delta;
            }
            dist(i, j) = sqrt (dist(i, j));
            dist(j, i) = dist(i, j);
        }
}

template <typename T>
inline T _abs_ (T x)
{
    return (x > T(0)) ? (x) : (-x);
}

template <typename T>
iv1 _abs_kth_largest_ (const vector<T>& x, int k = 1)
//  Given a vector x, return the ordered index (descent)
{
    int N = x.size ();
    vector<T> X = x;
    iv1 max_ind;    max_ind.assign (k, 0);
    for (int i = 0; i < k; i++)
    {
        T max = T(-100);
        for (int j = 0; j < N; j++)
            if (_abs_ (X[j]) > max)
            {
                max = _abs_ (X[j]);
                max_ind[i] = j;
            }
        X[max_ind[i]] = T(-200);
    }
}

double LATTICE::_periodic_ (dv1& dR_new, const dv1& dR, double Rc)
//  If dist > Rc, search if any of its periodic image has a dist <= Rc
//  store the new dR vector in dR_new, and return the new distance
//  Note that the solution is unique if a,b,c > 2*Rc
{
    int M = dR.size ();
    iv1 max_ind = _abs_kth_largest_ (dR, M);

    int trial = 0;
    while (trial < M)
    {
        trial ++;
        dv1 dRp = dR;
        for (int ind = 0; ind < trial; ind++)
        {
            int mu = max_ind[ind];
            int fac = (dRp[mu] > 0) ? (-1) : (1);
            dRp[mu] += fac * lat_len[mu];
        }
        double dist = 0.;
        for (int mu = 0; mu < M; mu++)  dist += dRp[mu] * dRp[mu];
        if (dist <= Rc * Rc)
        {
            dR_new = dRp;   return sqrt (dist);
        }
    }
    return -100.;
}

void LATTICE::_gen_neighbor_list_ (dv3& ndR, dv2& ndist,
    const vMatrixXd& dR, const MatrixXd& dist, double Rc)
{
    int N = dist.rows (), M = dR.size ();
    for (int i = 0; i < N; i++)
    {
        dv2 ndR0;   dv1 ndist0;
        for (int j = 0; j < N; j++)
        {
            if (i == j) continue;
            else if (dist (i, j) <= Rc)
            {
                ndist0.push_back (dist (i, j));
                dv1 tmp;
                for (int mu = 0; mu < M; mu++)  tmp.push_back (dR[mu](i, j));
                ndR0.push_back (tmp);
            }
            else
            {
                dv1 dR_new, dR_this;    dR_new.assign (M, 0.);
                for (int mu = 0; mu < M; mu++)  dR_this.push_back (dR[mu](i, j));
                double dist_new = _periodic_ (dR_new, dR_this, Rc);
                if (dist_new > 0)
                {
                    ndR0.push_back (dR_new);    ndist0.push_back (dist_new);
                }
            }
        }
        ndR.push_back (ndR0);   ndist.push_back (ndist0);
    }
}

inline double _cut_off_f_ (double R, double Rc)
{
    return (R > Rc) ? (0.) : (0.5 * (cos (M_PI * R / Rc) + 1.));
}

vvVectorXd LATTICE::_R_to_V_ (const vVectorXd& R, double Rc, const dv1& eta)
{
    int N = R.size (), M = R[0].size ();
//  generate dist matrix and directional dist matrices (dR)
    MatrixXd dist;  vMatrixXd dR;
    _gen_dist_mat_ (R, dR, dist);
//  generate neighboring element list
    dv2 ndist;  dv3 ndR;
    _gen_neighbor_list_ (ndR, ndist, dR, dist, Rc);

    vvVectorXd V;
    for (int i = 0; i < N; i++)
    {
//  iterate over atoms
        vVectorXd Vi;
        for (int mu = 0; mu < M; mu++)
        {
//  iterate over x, y, z
            VectorXd Vimu;   Vimu.setZero (eta.size ());
            for (int ind = 0; ind < eta.size (); ind++)
            {
//  iterate over components of eta
                double Vimu_eta = 0., et = eta[ind];
                for (int j = 0; j < ndR[i].size (); j++)
                {
//  iterate over neighbors
                    double r = ndist[i][j], x = ndR[i][j][mu];
                    Vimu_eta += exp (- (r * r / (et * et)))
                        * _cut_off_f_(r,Rc) * x / r;
                }
                Vimu[ind] = Vimu_eta;
            }
            Vi.push_back (Vimu);
        }
        V.push_back (Vi);
    }

    return V;
}

void LATTICE::_fingerprint_ ()
{
    for (int i = 0; i < R.size (); i++)
    {
        vvVectorXd V0 = _R_to_V_ (R[i], Rc, eta);
        V.insert (V.end (), V0.begin (), V0.end ());
    }
}

template <typename T>
void _shuffle_ (const iv1& ind, vector<T>& x)
{
    if (ind.size () != x.size ())
    {
        cout << "[Error] in _shuffle_: dimension does not match." << endl;
        exit (1);
    }

    vector<T> X(x.size ());
    for (int i = 0; i < x.size (); i++) X[i] = x[ind[i]];
    x = X;
}

void LATTICE::_shuffle_fingerprint_ ()
{
    iv1 indexes (V.size ());
    iota (indexes.begin (), indexes.end (), 0);
    random_shuffle (indexes.begin (), indexes.end ());

    _shuffle_ (indexes, V);
    _shuffle_ (indexes, F);
}

void LATTICE::_write_VF_ ()
{
    if (V.size () != F.size ())
    {
        cout << "[Error] In MLFFTRAIN: V and F must have the same size." << endl;
        exit (1);
    }

    string fnamex (out_path + "/fingerprint_x.dat");
    string fnamey (out_path + "/fingerprint_y.dat");
    string fnamez (out_path + "/fingerprint_z.dat");
    FILE *px = fopen (fnamex.c_str (), "w+");
    FILE *py = fopen (fnamey.c_str (), "w+");
    FILE *pz = fopen (fnamez.c_str (), "w+");
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
