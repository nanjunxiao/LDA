SRC_DIR=$(HOME)/ompi-lda

CXX=g++
CXXFLAGS=-O3 -fopenmp -I./include
#CXXFLAGS=-c -O2 -W  -I$(SRC_DIR)/common -I$(SRC_DIR)/utility -I/usr/include -I.

LDA:
	#$(CXX) $(CXXFLAGS) *.cc ./include/boost/program_options/src/*.cpp -o lda
	$(CXX) $(CXXFLAGS) *.cc ./include/boost/program_options/src/*.cpp -o infer 

.PHONY: clean
clean:
	#rm -f lda
