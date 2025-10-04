#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "machl_tensor.h"

Tensor* createTensor (int rows, int coll)
{
	Tensor *ten = malloc(sizeof (Tensor));

	ten -> rows = rows;
	ten -> coll = coll;
	ten -> ten = malloc (sizeof (float) * (rows * coll));

	return ten;
}

void resetTensor (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		ten -> ten[i] = 0;
}

void printTensor (Tensor *ten)
{
	int rows = ten -> rows;
	int coll = ten -> coll;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < coll; j++)
			printf ("%f ", ten -> ten[i * coll + j]);
		printf ("\n");
	}
	printf ("\n");
}

void uRandomTensor (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;

	srand (time (NULL));

	for (int i = 0; i < length; i++) {
		ten ->ten[i] = (float)rand() / RAND_MAX;
	}
}

void nRandomTensor (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;
	float sum;

	srand (time (NULL));

	for (int i = 0; i < length; i++) {
		sum = 0;
		for (int j = 0; j < 12; j++)
			sum += (float)rand() / RAND_MAX;
		ten -> ten[i] = sum - 6;
	}
}

void addScalar (Tensor *ten, float scalar)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		ten -> ten[i] += scalar;
}

void scaleTensor (Tensor *ten, float scalar)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		ten -> ten[i] *= scalar;
}

void transposeTensor (Tensor *ten, Tensor *tenT)
{
	int rows = ten -> rows;
	int coll = ten -> coll;
	int length = rows * coll;

	for (int i = 0; i < length; i++)
		tenT -> ten[i] = ten -> ten[(i % rows) * coll + i / rows];
}

void addTensors (Tensor *tenA, Tensor *tenB)
{
	int length = tenA -> rows * tenA -> coll;

	for (int i = 0; i < length; i++)
		tenA -> ten[i] += tenB -> ten[i];
}

float addMul (Tensor *tenL, Tensor *tenR, int row, int coll)
{
	int lColl = tenL -> coll;
	int rColl = tenR -> coll;
	float sum = 0;

	for (int i = 0; i < lColl; i++)
		sum += tenL -> ten[row * lColl + i] * tenR -> ten[i * rColl + coll];

	return sum;
}

void multiplyTensors (Tensor *tenL, Tensor *tenR, Tensor *res)
{
	int lRows = tenL -> rows;
	int rColl = tenR -> coll;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++)
			res -> ten[i * rColl + j] = addMull (tenL, tenR, i, j);
	}
}

void addMultiplyTensors (Tensor *tenL, Tensor *tenR, Tensor *res)
{
	int lRows = tenL -> rows;
	int rColl = tenR -> coll;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++)
			res -> ten[i * rColl + j] += addMull (tenL, tenR, i, j);
	}
}

void deleteTensor (Tensor *ten)
{
	if (ten == NULL)
		return;

	free (ten -> ten);
	free (ten);
	ten = NULL;
}
