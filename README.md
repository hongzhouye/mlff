# mlff
Machine learning force field generator implemented according to Botu _et. al._'s pioneering works (Ref. [1](http://onlinelibrary.wiley.com/doi/10.1002/qua.24836/abstract), [2](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.094306)).

# To compile
	make

# To run
	./mlff input/inp

# Input File

Lattice card
----------------------
|	Keyword	|	Type	|	Meaning	|
|---	|---		|---			|
|	Neta|	 int		|	Dimension of eta grid	|
|	eta	|	float;float	|	Uniformly generating a logarithmic-law grid.	|
|	lat_len	|	float;float;float	|	lattice const: _a_, _b_, _c_	|
|	lat_ang	|	float;float;float	|	lattice const: _alpha_, _beta_, _gamma_	|
|	Rc	|	float	|	cut off radius	|
|	shuffle	|	bool	|	shuffle the fingerprint or not	|
|	inp_path	|	string;string;string	|	_path_;_start_num_;_end_num_	|
|	write	|	bool	|	Writing the fingerprint files?	|
|	out_path	|	string	|	_path_ to write fingerprint files	|

Training card
--------------------------
|	Keyword	|	Type	|	Meaning	|
|---			|---		|---			|
|	Ntrain	|	int		|	training samples	|
|	k				|	int		|	_k_-fold cross validation	|
|	Nlbd		|	int		|	number of lambda's	|
|	lbd			|	float;float	|	lambda grid is generated in the same way as eta grid	|
	

SGD card
--------------------------
|	Keyword	|	Type	|	Meaning	|
|---			|---		|---			|
|	tau0		|	float	|	In formulat (tau0 + t)^(-kappa)	|
|	kappa		|	float	|	In formulat (tau0 + t)^(-kappa)	|
|	MAXITER	|	int		|	maximum number of iterations		|
|	Nbatch	|	int		|	size of batch for batch SGD			|

# Data File
	Format: x;y;z;Fx;Fy;Fz

# Output file
	Format: Vx(eta_1);Vx(eta_2);...;V(eta_Neta);Fx
