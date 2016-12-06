eig_path=${PWD}

CXX=g++-6
CXXFLAGS=-Ofast -std=c++11 -w -msse2 -mavx -mfma -fopenmp #-framework Accelerate
CXXINC=${eig_path}

mlff: main.o krr.o mlfftrain.o lattice.o read.o utils.o #normdist.o
	$(CXX) -I$(CXXINC) $(CXXFLAGS) main.o krr.o mlfftrain.o lattice.o\
		read.o utils.o -o mlff
		#normdist.o

main.o: main.cpp include/utils.hpp include/mlfftrain.hpp include/read.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) main.cpp -c

krr.o: krr.cpp include/krr.hpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) krr.cpp -c

mlfftrain.o: mlfftrain.cpp include/mlfftrain.hpp include/krr.hpp include/lattice.hpp\
	include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) mlfftrain.cpp -c

lattice.o: lattice.cpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) lattice.cpp -c

#normdist.o: normdist.cpp include/normdist.hpp include/utils.hpp
#	$(CXX) -I$(CXXINC) $(CXXFLAGS) normdist.cpp -c

read.o: read.cpp include/read.hpp include/utils.hpp include/mlfftrain.hpp\
	include/lattice.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) read.cpp -c

utils.o: utils.cpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) utils.cpp -c

clean:
	rm -f *.o
