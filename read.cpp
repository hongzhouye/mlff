#include "include/read.hpp"

/* These two functions split a string with given delimeter */
void _split_ (const string &s, char delim, vector<string> &elems) {
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

vector<string> _split_ (const string &s, char delim) {
    vector<string> elems;
    _split_ (s, delim, elems);
    return elems;
}

/* This function turns a string into all uppercase */
string _uppercase_ (const string& s)
{
	string result (s.length (), ' ');
	transform (
			s.begin (),
			s.end (),
			result.begin (),
			ptr_fun <int, int> (toupper)
			);
	return result;
}

/* split a string like
 *      "N = 5" or "N=5" or "N= 5" or "N =5"
 * into a vector<string> with elements "N" and "5"
 */
vs _split_eq_ (string line)
{
    vs temp = _split_ (line, '=');

    vs namestr = _split_ (temp[0], ' ');
	string name = *(namestr.begin ());

	vs valuestr = _split_ (temp[1], ' ');
	string value = valuestr[valuestr.size () - 1];

    vs nv;
    nv.push_back (name);    nv.push_back (value);

    return nv;
}

void _read_inp_ (string& fname, LATTICE& lat, MLFFTRAIN& fft)
{
    string line;
    vs spline;
    ifstream input (fname);

    if (input)
    {
        while (getline (input, line))
        {
            if (line.empty () || line[0] == '#')    continue;

            if (_uppercase_ (line) == "&LATTICE")
            {
                while (getline (input, line))
                {
                    if (line.empty () || line[0] == '#')    continue;
                    if (_uppercase_ (line) == "&END")   break;

                    spline = _split_eq_ (line);
                    if (_uppercase_ (spline[0]) == "NETA")
                    {
                        lat.Neta = (int) stod (spline[1]);
                        lat.eta.assign (fft.Neta, 0.);
                    }
                    else if (_uppercase_ (spline[0]) == "ETA")
                    {
                        vs temp = _split_ (spline[1], ';');
                        _log_space_ (lat.eta, lat.Neta,
                            stod (temp[0]), stod (temp[1]));
                    }
                    else if (_uppercase_ (spline[0]) == "LAT_LEN")
                    {
                        vs temp = _split_ (spline[1], ';');
                        for (int mu = 0; mu < temp.size (); mu++)
                            lat.lat_len.push_back (stod (temp[mu]));
                    }
                    else if (_uppercase_ (spline[0]) == "LAT_ANG")
                    {
                        vs temp = _split_ (spline[1], ';');
                        for (int mu = 0; mu < temp.size (); mu++)
                            lat.lat_ang.push_back (stod (temp[mu]));
                    }
                    else if (_uppercase_ (spline[0]) == "RC")
                        lat.Rc = stod (spline[1]);
                    else if (_uppercase_ (spline[0]) == "INP_PATH")
                    {
                        vs temp = _split_ (spline[1], ';');
                        _read_data_ (
                            temp[0], stoi (temp[1]), stoi (temp[2]), lat);
                    }
                    else if (_uppercase_ (spline[0]) == "WRITE")
                        lat.write = (_uppercase_ (spline[1]) == "TRUE") ?
                            (true) : (false);
                    else if (_uppercase_ (spline[0]) == "OUT_PATH")
                        lat.out_path = spline[1];
                    else if (_uppercase_ (spline[0]) == "SHUFFLE")
                        lat.shuf = (_uppercase_ (spline[1]) == "TRUE") ?
                            (true) : (false);
                    else
                    {
                        cout << "[Error] unknown mode: " << line << endl
                            << "Please check the LATTICE card." << endl;
                        exit (1);
                    }
                }
            }
            else if (_uppercase_ (line) == "&TRAINING")
            {
                while (getline (input, line))
                {
                    if (line.empty () || line[0] == '#')    continue;
                    if (_uppercase_ (line) == "&END")   break;

                    spline = _split_eq_ (line);
                    if (_uppercase_ (spline[0]) == "NTEST")
                        fft.Ntest = (int) stod (spline[1]);
                    else if (_uppercase_ (spline[0]) == "NTRAIN")
                        fft.Ntrain = (int) stod (spline[1]);
                    else if (_uppercase_ (spline[0]) == "K")
                        fft.K = (int) stod (spline[1]);
                    else if (_uppercase_ (spline[0]) == "GAMMA")
                        fft.gamma = stod (spline[1]);
                    else if (_uppercase_ (spline[0]) == "NLBD")
                    {
                        fft.Nlbd = (int) stod (spline[1]);
                        fft.lbd_set.assign (fft.Nlbd, 0.);
                    }
                    else if (_uppercase_ (spline[0]) == "LBD")
                    {
                        vs temp = _split_ (spline[1], ';');
                        _log_space_ (fft.lbd_set, fft.Nlbd,
                            stod (temp[0]), stod (temp[1]));
                    }
                    else
                    {
                        cout << "[Error] unknown mode: " << line << endl
                            << "Please check the TRAINING card." << endl;
                        exit (1);
                    }
                }
            }
            else
            {
                cout << "[Error] unknown card name: " << line << endl;
                exit (1);
            }
        }
    }
    else
    {
        cout << "[Error] cannot open file: " << fname << endl;
        exit (1);
    }
    input.close ();
}

void _read_data_ (string& path, int start, int end, LATTICE& lat)
{
    string line;
    vs fname_set, spline;
    for (int i = start; i <= end; i++)
    {
        string fname (path + "/" + "data" + to_string (i) + ".dat");
        fname_set.push_back (fname);

        vVectorXd R, F;

        ifstream input (fname);
        if (input)
        {
            while (getline (input, line))
            {
                if (line.empty ())  break;

                spline =  _split_ (line, ';');
                if (spline.size () != 6)
                {
                    cout << "[Error] please check data file: " << fname << endl;
                    exit (1);
                }

                VectorXd r(3), f(3);
                for (int i = 0; i < 3; i++)
                {
                    r(i) = stod (spline[i]);
                    f(i) = stod (spline[i + 3]);
                }
                R.push_back (r);
                F.push_back (f);
            }
        }
        else
        {
            cout << "[Error] cannot open file: " << fname << endl;
            exit (1);
        }
        input.close ();

        if (R.size () > 0)
        {
            lat.R.push_back (R);    // to be convert to fignerprint
            lat.F.insert (lat.F.end (), F.begin (), F.end ());
        }
    }
}
