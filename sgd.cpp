#include <iostream>
#include <fstream>
#include <vector>

#include <cmath>

using namespace std;


/* IO Functions */
int32_t read32_he(std::istream& stream)
{
    uint8_t buf[4];
    stream.read((char*)buf, 4);
    
    return static_cast<int32_t>(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
}


uint8_t** loadImgFile(const string &path, int *len, int *rows, int *cols)
{
	ifstream inFile;
	int magic = 0;
	uint8_t **ret;

	inFile.open(path, ios::in|ios::binary);

	magic = read32_he(inFile);
	*len = read32_he(inFile);

	if (magic != 2051)
		throw runtime_error("Unexpected magic in file");

	*rows = read32_he(inFile);
	*cols = read32_he(inFile);

	ret = new uint8_t*[*len];

	for (int i = 0; i < *len; i++) {
		ret[i] = new uint8_t[*rows * *cols];
		inFile.read((char*)ret[i], *rows * *cols);
	}

	inFile.close();
	return ret;
}

uint8_t* loadLabelFile(const string &path, int *len)
{
	ifstream inFile;
	int magic = 0;
	uint8_t *ret;

	inFile.open(path, ios::in|ios::binary);

	magic = read32_he(inFile);
	*len = read32_he(inFile);

	if (magic != 2049)
		throw runtime_error("Unexpected magic in file");

	ret = new uint8_t[*len];

	for (int i = 0; i < *len; i++) {
		inFile.read((char*)&ret[i], 1);
	}

	inFile.close();
	return ret;
}

void printImg(uint8_t *img, int rows, int cols)
{
	for (int i = 0; i < rows * cols; i++) {
		if (i % 28 == 0) {
			printf("\n");
			continue;
		}
		if (img[i] > 150)
			printf("x");
		else
			printf(" ");
	}
	printf("\n");
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
uint8_t* gradientDescent(uint8_t** X, uint8_t *y, const size_t m, const size_t n)
{
	uint8_t *theta = new uint8_t[m];
}



int main(int argc, const char *argv[])
{
	int lenTest = 0;
	int lenTrain = 0;
	int rowsTrain;
	int colsTrain;
	int rowsTest;
	int colsTest;

	uint8_t* testLabels = loadLabelFile("t10k-labels-idx1-ubyte", &lenTest);
	uint8_t** testImages = loadImgFile("t10k-images-idx3-ubyte", &lenTest, &rowsTest, &colsTest);
	uint8_t* trainLabels = loadLabelFile("train-labels-idx1-ubyte", &lenTrain);
	uint8_t** trainImages = loadImgFile("t10k-images-idx3-ubyte", &lenTrain, &rowsTrain, &colsTrain);


	for (int i = 0; i < 10; i++) {
		printImg(testImages[i], rowsTest, colsTest);
	}

	return 0;
}
