/* Includes, system */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Vector size */
#define N  (4096)

__global__ void d_apply_sigmoid(float *r, int l)
{
	int index = threadIdx.x;
	if (index < l) {
		float val = r[index];
		r[index] = 1.0 / (1.0 + exp(-val));
	}
}

__global__ void d_subs(float *A, float *B, float *C)
{
	A[threadIdx.x] = B[threadIdx.x] - C[threadIdx.x];
}

void d_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


cublasStatus_t init_cublas(cublasHandle_t *handle)
{
	/* Initialize CUBLAS */
	fprintf(stderr, "simpleCUBLAS test running..\n");
	cublasStatus_t status = cublasCreate(handle);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Failed to initialize cublas\n");
		fprintf(stderr, "Error number: %d\n", status);
		exit(1);
	}
	return status;
}

/* Main */
int main(int argc, char **argv)
{
    cublasHandle_t handle;
		init_cublas(&handle);

#if FALSE
    float *h_X, *h_Y, *h_res;
    float *d_X = 0;
    float *d_Y = 0;

    float *d_res = 0;

    int i;


    /* Allocate host memory for the matrices */
    h_X = (float *)malloc(N * sizeof(h_X[0]));
    h_Y = (float *)malloc(N * sizeof(h_Y[0]));
    h_res = (float *)malloc(N * sizeof(h_res[0]));

    /* Fill the matrices with test data */
    for (i = 0; i < N; i++)
    {
        h_X[i] = 2.0;
        h_Y[i] = 2.0;
    }

    /* Allocate device memory for the matrices */
    cudaMalloc((void **)&d_X, N * sizeof(d_X[0]));
    cudaMalloc((void **)&d_Y, N * sizeof(d_Y[0]));
    cudaMalloc((void **)&d_res, N * sizeof(d_res[0]));
    fprintf(stderr, "malloc\n");

    /* Initialize the device matrices with the host matrices */
    cublasSetVector(N, sizeof(h_X[0]), h_X, 1, d_X, 1);
    cublasSetVector(N, sizeof(h_Y[0]), h_Y, 1, d_Y, 1);
		cublasSetVector(N, sizeof(h_res[0]), h_res, 1, d_res, 1);
    fprintf(stderr, "setVector\n");

    /* Performs operation using cublas */
		cublasSdot(handle, N, d_X, 1, d_Y, 1, h_res);
    fprintf(stderr, "sDot\n");

    /* Read the result back */
    //cublasGetVector(N, sizeof(h_res[0]), d_res, 1, h_res, 1);
    //printf("getVector\n");

		int m = 60000;
		int n = (28 * 28 + 1);
#endif

		int X_rows = 60000;
		int X_cols = 784;

		/* actually transposed theta', which makes
		 * setting parameter below easier
		 */
		int theta_rows = X_rows;
		int theta_cols = 1;
		int result_rows = X_rows;
		int result_cols = theta_cols;
		int y_rows = theta_cols;
		int y_cols = theta_rows;

		float *X = (float *)malloc(sizeof(float) * X_rows * X_cols);
		float *d_X;
		float *theta = (float *)malloc(sizeof(float) * theta_rows * theta_cols);
		float *d_theta;
		// Multiplication result
		float *result = (float *)malloc(sizeof(float) * result_rows * result_cols);
		float *d_result;
		// Ys
		float *y = (float *) malloc(sizeof(float) * y_cols);
		float *d_y;

		for (int i = 0; i < X_cols; i++) {
			for (int j = 0; j < X_rows; j++) {
				X[i + j] = i * j;
			}
			y[i] = i;
			theta[i] = 0;
		}

		cudaMalloc((void **)&d_X, X_rows * X_cols * sizeof(float));
		cudaMalloc((void **)&d_theta, theta_rows * theta_cols * sizeof(float));
		cudaMalloc((void **)&d_result, result_rows * result_cols * sizeof(float));
		cudaMalloc((void **)&d_y, y_rows * y_cols * sizeof(float));

		/* 
		 * X = {1 1  t = {1
		 *      2 1}      1}
		 */
		cudaMemcpy(d_X, X, sizeof(*X) * X_rows * X_cols, cudaMemcpyHostToDevice);
		cudaMemcpy(d_theta, theta, sizeof(*theta) * theta_rows * theta_cols, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, sizeof(*y) * y_rows * y_cols, cudaMemcpyHostToDevice);

		for (int i = 0; i < 1000; i++) {
			fprintf(stderr, "[%d/%d]\n", i + 1, 100);

			/* Setting these is not trivial:
			 * more info here: http://docs.nvidia.com/cuda/cublas/index.html#cublassetmatrix
			 */
			d_mmul(handle, d_X, d_theta, d_result, X_rows, theta_rows, X_cols);

			d_apply_sigmoid<<<1, result_cols * result_rows>>>(d_result, result_cols * result_rows);

			d_subs<<<1, y_cols>>>(d_result, d_result, d_y);

			float beta = 0;
			float alpha = 1;
								/*handle  transa       transb         m          n         k      alpha   A   lda       B         ldb        beta     C         ldc */ 
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, X_cols, result_cols, X_rows, &alpha, d_X, X_cols, d_result,  X_rows,     &beta,  d_result,  X_cols);
		}


		cudaMemcpy(result, d_result, sizeof(*result) * result_rows * result_cols, cudaMemcpyDeviceToHost);


}
