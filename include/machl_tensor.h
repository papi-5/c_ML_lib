#ifndef MACHL_TENSOR_H
#define MACHL_TENSOR_H

struct tensor {
	float *ten;
	int rows;
	int coll;
};

typedef struct tensor Tensor;

Tensor* createTensor (int rows, int coll);

int sizeOfTensor (Tensor *ten);		// returns size in bytes

void resetTensor (Tensor *ten);		// sets all elements to 0

void printTensor (Tensor *ten);

void uRandomTensor (Tensor *ten);	// randomizes from U(0, 1)

void nRandomTensor (Tensor *ten);	// randomizes from N(0, 1)

void addScalar (Tensor *ten, float scalar);

void scaleTensor (Tensor *ten, float scalar);

void transposeTensor (Tensor *ten, Tensor *tenT);

void addTensors (Tensor *tenA, Tensor *tenB);

float addMull (Tensor *tenL, Tensor *tenR, int row, int coll);

void multiplyTensors (Tensor *tenL, Tensor *tenR, Tensor *res);

void addMultiplyTensors (Tensor *tenL, Tensor *tenR, Tensor *res);

void deleteTensor (Tensor *ten);

#endif
