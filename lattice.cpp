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
                double delta = R[j](mu) - R[i](mu);
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
            if (_abs_ (X[j]) > max && _abs_ (X[j]) < 200)
            {
                max = _abs_ (X[j]);
                max_ind[i] = j;
            }
        X[max_ind[i]] = T(-200);
    }
	return max_ind;
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
                        * _cut_off_f_(r, Rc) * x / r;
                }
                Vimu(ind) = Vimu_eta;
            }
            Vi.push_back (Vimu);
        }
        V.push_back (Vi);
    }

    return V;
}

void LATTICE::_print_ ()
{
    cout << "#eta grid:" << endl;
    for (int i = 0; i < Neta; i++)  printf ("%5.3e  ", eta[i]);
    cout << endl;
}

void LATTICE::_fingerprint_ (const vvVectorXd& R, vvVectorXd& V, vVectorXd& F)
{
    for (int i = 0; i < R.size (); i++)
    {
        vvVectorXd V0 = _R_to_V_ (R[i], Rc, eta);
        V.insert (V.end (), V0.begin (), V0.end ());
    }
    if (shuf)   _shuffle_fingerprint_ (V, F);
    if (write)  _write_VF_ ();
}

void LATTICE::_fingerprint_ (const vvVectorXd& R, vvVectorXd& V)
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

//template <typename T1, typename T2>
//void LATTICE::_shuffle_fingerprint_ (vector<T1>& V, vector<T2>& F)
void LATTICE::_shuffle_fingerprint_ (vvVectorXd& V, vVectorXd& F)
{
    iv1 indexes (V.size ());
    iota (indexes.begin (), indexes.end (), 0);
    srand(time(0));
    random_shuffle (indexes.begin (), indexes.end ());

    _shuffle_ (indexes, V);
    _shuffle_ (indexes, F);
}

void LATTICE::_shuffle_fingerprint_ (vVectorXd& V, dv1& F)
{
    iv1 indexes (V.size ());
    iota (indexes.begin (), indexes.end (), 0);
    srand(time(0));
    random_shuffle (indexes.begin (), indexes.end ());

    _shuffle_ (indexes, V);
    _shuffle_ (indexes, F);
}

dv1 _count_ (const dv2& dist, double Rc, const dv1& bin)
{
    int N = dist.size (), nbin = bin.size ();
    double dr = Rc / (nbin - 1);
    dv1 rdf;
    for (int I = 0; I < nbin - 1; I++)
    {
        iv1 count;  count.assign (N, 0);
        for (int i = 0; i < N; i++)
            for (auto j = dist[i].begin (); j < dist[i].end (); j++)
                if (*j >= bin[I] && *j < bin[I + 1])  count[i] ++;
        int sum = 0;
        for (int n : count)    sum += n;
        rdf.push_back (sum / (double) N);
    }
    return rdf;
}

dv1 LATTICE::_rdf_per_cell_ (const vVectorXd& R, double Rc, const dv1& bin)
{
    int N = R.size (), M = R[0].size ();
//  generate dist matrix and directional dist matrices (dR)
    MatrixXd dist;  vMatrixXd dR;
    _gen_dist_mat_ (R, dR, dist);
//  generate neighboring element list
    dv2 ndist;  dv3 ndR;
    _gen_neighbor_list_ (ndR, ndist, dR, dist, Rc);
    dv1 rdf = _count_ (ndist, Rc, bin);
    return rdf;
}

dv1 _gen_mesh_ (double min, double max, int n)
{
    dv1 x;  double dx = (max - min) / (n - 1.);
    for (int i = 0; i < n; i++) x.push_back (dx * i);
    return x;
}

void LATTICE::_gen_rdf_ (int nbin)
{
    dv1 bin = _gen_mesh_ (0., Rc, nbin);
    dv1 rdf_per_cell, rdf;  rdf.assign (nbin, 0.);
    for (int i = 0; i < R.size (); i++)
    {
        rdf_per_cell = _rdf_per_cell_ (R[i], Rc, bin);
        //cout << "rdf.size = " << rdf_per_cell.size () << endl;
        for (int j = 0; j < nbin; j++)  rdf[j] += rdf_per_cell[j];
    }
    for (int j = 0; j < nbin; j++)
    {
        rdf[j] /= (double) R.size ();
        printf ("%5.3f\t%9.6f\n", bin[j], rdf[j]);
        fflush (0);
    }
}

iv2 LATTICE::_form_index_ (const vvVectorXd& V)
{
    int N = V.size (), M = V[0].size ();

    iv2 ind_set;
    for (int mu = 0; mu <= M; mu++) {iv1 a; ind_set.push_back (a);}
    iv1 num_zeros;  num_zeros.assign (M + 1, 0);
    int count_zero;
    for (int i = 0; i < N; i++)
    {
        count_zero = 0;
        for (int mu = 0; mu < M; mu++)
            if (V[i][mu].norm () < 1.E-5)   count_zero++;
        ind_set[count_zero].push_back (i);
        num_zeros[count_zero] ++;
    }
    for (int mu = 0; mu <= M; mu++)
        cout << "# of having " << mu << " zeros: " << num_zeros[mu] << endl;
    cout << endl;

    return ind_set;
}

void LATTICE::_form_sanity_set_ (
    vvVectorXd& Vtrain, vVectorXd& Ftrain,
    vvVectorXd& Vtest, vVectorXd& Ftest, int Nsanity)
{
    int N = V.size (), M = V[0].size ();

    iv2 index_set = _form_index_ (Vsanity);
    for (int mu = 0; mu < M; mu++)
        if (2 * Nsanity >= index_set[mu].size ())
            Nsanity = (int) index_set[mu].size () / 2;
    cout << "Nsanity = " << Nsanity << endl;

    for (int mu = 0; mu < M; mu++)
    {
        for (int i = 0; i < Nsanity; i++)
        {
            int ind = index_set[mu][i];
            Vtrain.push_back (Vsanity[ind]);
            Ftrain.push_back (Fsanity[ind]);
            ind = index_set[mu][i + Nsanity];
            Vtest.push_back (Vsanity[ind]);
            Ftest.push_back (Fsanity[ind]);
        }
    }

    int ind = index_set[M][0];
    Vtrain.push_back (Vsanity[ind]);    Ftrain.push_back (Fsanity[ind]);
    ind = index_set[M][1];
    Vtest.push_back (Vsanity[ind]);    Ftest.push_back (Fsanity[ind]);
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
