eig_path=${PWD}

CXX=g++
CXXFLAGS=-Ofast -std=c++11 -w
CXXINC=${eig_path}

mlff: main.o sgd.o mlfftrain.o lattice.o normdist.o read.o utils.o
	$(CXX) -I$(CXXINC) $(CXXFLAGS) main.o sgd.o mlfftrain.o lattice.o normdist.o \
		read.o utils.o -o mlff

main.o: main.cpp include/utils.hpp include/mlfftrain.hpp include/read.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) main.cpp -c

sgd.o: sgd.cpp include/sgd.hpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) sgd.cpp -c

mlfftrain.o: mlfftrain.cpp include/mlfftrain.hpp include/sgd.hpp include/lattice.hpp\
	include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) mlfftrain.cpp -c

lattice.o: lattice.cpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) lattice.cpp -c

normdist.o: normdist.cpp include/normdist.hpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) normdist.cpp -c

read.o: read.cpp include/read.hpp include/utils.hpp include/mlfftrain.hpp include/sgd.hpp\
	include/lattice.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) read.cpp -c

utils.o: utils.cpp include/utils.hpp
	$(CXX) -I$(CXXINC) $(CXXFLAGS) utils.cpp -c

clean:
	rm -f *.o
