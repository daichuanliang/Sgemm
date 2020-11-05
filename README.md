# Sgemm
Optimization of matrix multiplication

[How to build]
	g++ -O3 -std=c++11 sgemm.cpp -mfma -o sgemm
	g++ -O3 -std=c++11 transSgemm.cpp -mavx2 -o transSgemm
