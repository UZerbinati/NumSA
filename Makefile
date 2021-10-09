CXX=g++ -std=c++14
all: python
python:

	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./fem.cpp -fopenmp -o py/numsa/fem`python3-config --extension-suffix`
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./nla.cpp -fopenmp -o py/numsa/NLA`python3-config --extension-suffix`
