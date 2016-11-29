#define _XOPEN_SOURCE

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <math.h>


/* IO Functions */
int32_t read32_he(FILE *f)
{
    uint8_t buf[4];
		fread((char*)buf, 4, 1, f);
    
    return (int32_t)(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
}


uint8_t** load_img_file(const char *path, int *len, int *rows, int *cols)
{
	FILE *f;
	int magic = 0;
	uint8_t **ret;

	f = fopen(path, "r");

	magic = read32_he(f);
	*len = read32_he(f);

	assert(magic == 2051);

	*rows = read32_he(f);
	*cols = read32_he(f);

	ret = malloc(sizeof(uint8_t*) * (*len + 1));

	for (int i = 0; i < (*len + 1); i++) {
		ret[i] = malloc(*rows * *cols);
		if (i == 0)
			ret[i][0] = (uint8_t)1; /* Set our first feature to 1 */
		fread(ret[i], sizeof(uint8_t), *rows * *cols, f);
	}

	fclose(f);
	return ret;
}

uint8_t* load_label_file(const char *path, int *len)
{
	FILE *f;
	int magic = 0;
	uint8_t *ret;

	f = fopen(path, "r");

	magic = read32_he(f);
	*len = read32_he(f);

	assert(magic == 2049);

	ret = malloc(sizeof(uint8_t*) * *len);

	for (int i = 0; i < *len; i++) {
		fread(&ret[i], 1, 1, f);
	}

	fclose(f);
	return ret;
}

void print_img(uint8_t *img, int rows, int cols, FILE *stream)
{
	for (int i = 0; i < rows * cols; i++) {
		if (i % 28 == 0) {
			fprintf(stream, "\n");
			continue;
		}
		if (img[i] > 150)
			fprintf(stream, "x");
		else
			fprintf(stream, " ");
	}
	fprintf(stream, "\n");
}


/* Machine learning stuff */
double sigmoid(double x)
{
	double exp_value;
	double return_value;

	exp_value = exp((double) -x);

	return_value = 1 / (1 + exp_value);

	return return_value;
}

/* computes logistic gradientDescent using
 * an matrix X[m][n] that consists of
 * m training examples by n features
 */
double hypothesis(uint8_t *x, double* theta, int n){
	/* h_theta_x = 1 / (1 + e^ - theta' X) */

	double logit = 0.0;
	for(int i = 0; i < n; i++){
		logit += x[i] * theta[i];
	}
	return sigmoid(logit);
}


double cost_function(uint8_t **X, uint8_t *y, double *theta, int m, int n)
{
	double error = 0.0;
	for (int i = 0; i < m; i++) {
		double hi = hypothesis(X[i], theta, n);
		if (y[i] == 1)
			error += y[i] * log(hi);
		else
			error += (1 - y[i]) * log(1 - hi);
	}
	double J = -1/m * error;
	return J;
}

double cost_function_prime(uint8_t **X, uint8_t *y, double *theta, int j, int m, int n, double alpha)
{
	double error = 0.0;
	for (int i = 0; i < m; i++) {
		double hi = hypothesis(X[i], theta, n);
		error += (hi - y[i]) * X[i][j];
	}
	double J = alpha/m * error;
	return J;
}


void gradient_descent(uint8_t **X, uint8_t *y, double *theta, int m, int n, double alpha)
{
	for (int j = 0; j < n; j++) {
		theta[j] = theta[j] - cost_function_prime(X, y, theta, j, m, n, alpha);
	}
}

void logistic_regression(uint8_t **X, uint8_t *y, double *theta, int m, int n, double alpha, int iters)
{
	for (int x = 0; x < iters; x++) {
		fprintf(stderr, "iter: [%d/%d]\n", x + 1, iters);
		gradient_descent(X, y, theta, m, n, alpha);
	}
}

void pretty(uint8_t** images, uint8_t *labels, int how_many, int rows, int cols)
{
	fprintf(stderr, "Printing a couple of numbers and its (given) labels\n");
	for (int i = 0; i < how_many; i++) {
		print_img(images[i], rows, cols, stderr);
		fprintf(stderr, "%d\n", labels[i]);
		fprintf(stderr, "====================\n");
	}
}

int main(int argc, char **argv)
{
	int lenTest = 0;
	int lenTrain = 0;
	int rowsTrain;
	int colsTrain;
	int rowsTest;
	int colsTest;


  bool print = 0;
	int sgd_iter = 5;
  int c;

  opterr = 0;

	while ((c = getopt (argc, argv, "pi:")) != -1) {
		switch (c)
		{
			case 'p':
				print = 1;
				break;
			case 'i':
				sgd_iter = atoi(optarg);
				break;
			case '?':
				if (optopt == 'c')
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				abort ();
		}
	}



	uint8_t* testLabels = load_label_file("t10k-labels-idx1-ubyte", &lenTest);
	uint8_t** testImages = load_img_file("t10k-images-idx3-ubyte", &lenTest, &rowsTest, &colsTest);
	uint8_t* trainLabels = load_label_file("train-labels-idx1-ubyte", &lenTrain);
	uint8_t** trainImages = load_img_file("t10k-images-idx3-ubyte", &lenTrain, &rowsTrain, &colsTrain);

	if (print) {
		pretty(testImages, testLabels, 5, rowsTest, colsTest);
		return EXIT_SUCCESS;
	}

	int n = (rowsTrain * colsTrain) + 1; // number of features; we just use each 'pixel' as a feature, and that matrices
					       											 // have length n + 1
	int m = lenTrain;                    // number of training examples
	uint8_t **X = trainImages;           // input features
	uint8_t *y = trainLabels;            // target variable
	double alpha = 0.001;

	double *theta = (double*) calloc(sizeof(double), n);

	logistic_regression(X, y, theta, m, n, alpha, sgd_iter);

	for (int i =0; i < n; i++)
		printf("%f\n", theta[i]);

	return 0;
}
