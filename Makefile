CXX=g++ -std=c++14
all: python
python:

	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./core.cpp -fopenmp -o py/numsa/core`python3-config --extension-suffix`
mesh.o: 
	$(CXX) -c core/mesh.cpp -o Build/mesh.o
space.o: 
	$(CXX) -c core/space.cpp -o Build/space.o
