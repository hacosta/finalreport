all: sgd

sgd: sgd.cpp
	g++ -Wall -O3 -std=c++11 sgd.cpp -o sgd 

clean:
	rm sgd
