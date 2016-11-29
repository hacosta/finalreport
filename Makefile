all: sgd cudasgd cudatest dot

sgd: sgd.c
	gcc -std=c99 -Wall -lm -O3 sgd.c -o sgd 

cudasgd: cudasgd.cu
	nvcc  -arch=compute_35 -code=sm_35  cudasgd.cu -o cudasgd

#cudatest: cudatest.cu
#	nvcc  -arch=compute_35 -code=sm_35  cudatest.cu -o cudatest

# Test
dot: dot.cu
	nvcc  -lcublas -arch=compute_35 -code=sm_35  dot.cu -o dot

clean:
	rm -f sgd cudasgd cudatest
