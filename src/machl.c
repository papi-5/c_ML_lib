#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "machl.h"

ActFunction activationFunctions[] = {{sigmoid, derSigmoid},
					{Tanh, derTanh},
					{RElu, derRElu},
					{softmax, derSoftmax}};

CostFunction costFunctions[] = {{MSE, derMSE},
				{crossEntropy, derCrossEntropy}};

void sigmoid (Matrix *mat)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++)
		(mat -> mat)[i] = 1.0 / (1.0 + exp ((mat -> mat)[i] * -1));
}

void derSigmoid (Matrix *mat, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		(res -> mat)[i] = x * (1.0 - x);
	}
}

void Tanh (Matrix *mat)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		double tmp1 = exp (x);
		double tmp2 = exp (-x);
		(mat -> mat)[i] = (tmp1 - tmp2) / (tmp1 + tmp2);
	}
}

void derTanh (Matrix *mat, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = 1.0 - pow ((mat -> mat)[i], 2.0);
}

void RElu (Matrix *mat)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++) {
		if ((mat -> mat)[i] <= 0)
			(mat -> mat)[i] = 0;
	}
}

void derRElu (Matrix *mat, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++) {
		if ((mat -> mat)[i] > 0)
			(res -> mat)[i] = 1;
		else
			(res -> mat)[i] = 0;
	}
}

void softmax (Matrix *mat)
{
	int length = mat -> coll * mat -> rows;
	double sum = 0;
	Matrix *tmp = createMatrix (1, length);

	for (int i = 0; i < length; i++) {
		(tmp -> mat)[i] = exp ((mat -> mat)[i]);
		sum += (tmp -> mat)[i];
	}

	for (int i = 0; i < length; i++)
		(mat -> mat)[i] = (tmp -> mat)[i] / sum;

	deleteMatrix (tmp);
}

void derSoftmax (Matrix *mat, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		(res -> mat)[i] = x * (1.0 - x);
	}
}

double MSE (Matrix *mat, Matrix *y)
{
	int length = mat -> coll * mat -> rows;
	double res = 0;

	for (int i = 0; i < length; i++) {
		double tmp = (mat -> mat)[i] - (y -> mat)[i];
		res +=	pow (tmp, 2.0);
	}

	res /= length;

	return res;
}

void derMSE (Matrix *mat, Matrix *y, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = 2.0 * ((mat -> mat)[i] - (y -> mat)[i]);
}

double crossEntropy (Matrix *mat, Matrix *y)
{
	int length = mat -> coll * mat -> rows;
	double res = 0;

	for (int i = 0; i < length; i++)
		res += (y -> mat)[i] * log ((mat -> mat)[i]);

	res *= -1.0;

	return res;
}

void derCrossEntropy (Matrix *mat, Matrix *y, Matrix *res)
{
	int length = mat -> coll * mat -> rows;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = -1.0 * ((y -> mat)[i] / (mat -> mat)[i]);
}



void printMatrix (Matrix *mat)
{
	int rows = mat -> rows;
	int coll = mat -> coll;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < coll; j++) {
			printf ("%lf ", (mat -> mat)[i * coll + j]);
		}
		printf("\n");
	}
	printf("\n");
}

Matrix* createMatrix (int rows, int coll)
{
	Matrix *mat = malloc (sizeof (Matrix));

	mat -> rows = rows;
	mat -> coll = coll;
	mat -> mat = calloc (rows * coll, sizeof (double));

	return mat;
}

void randomMatrix (Matrix *mat, double lowerBound, double upperBound)
{
	int rows = mat -> rows;
	int coll = mat -> coll;

	srand (time (NULL));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < coll; j++) {
			(mat -> mat)[i * coll + j] = ((double)rand() / RAND_MAX) * (upperBound - lowerBound) + lowerBound;
		}
	}
}

Matrix* createRandomMatrix (int rows, int coll, double lowerBound, double upperBound)
{
	Matrix *mat = createMatrix (rows, coll);
	randomMatrix (mat, lowerBound, upperBound);

	return mat;
}

void resetMatrix (Matrix *mat)
{
	int length = mat -> rows * mat -> coll;

	for (int i = 0; i < length; i++)
		mat -> mat[i] = 0;
}

void transposeMatrix (Matrix *mat, Matrix *matT)
{
	int rows = mat -> rows;
	int coll = mat -> coll;

	if (rows != matT -> coll
		|| coll != matT -> rows)
		return;

	for (int i = 0; i < rows * coll; i++) {
		(matT -> mat)[i] = (mat -> mat)[(i % rows) * coll + i / rows];
	}
}

void scaleMatrix (Matrix *mat, double factor)
{
	int length = mat -> rows * mat -> coll;

	for (int i = 0; i < length; i++)
		mat -> mat[i] *= factor;
}

void addMatrices (Matrix *matA, Matrix *matB)
{
	int length = matA -> rows * matA -> coll;

	for (int i = 0; i < length; i++)
		(matA -> mat)[i] += (matB -> mat)[i];
}

double matrixMultiplication (Matrix *left, Matrix *right, int row, int coll)
{
	int length = left -> coll;
	int lColl = left -> coll;
	int rColl = right -> coll;
	double *lMat = left -> mat;
	double *rMat = right -> mat;
	double sum = 0;

	for (int i = 0; i < length; i++) {
		sum += lMat[row * lColl + i] * rMat[i * rColl + coll];
	}

	return sum;
}

void multiplyMatrices (Matrix *left, Matrix *right, Matrix *result)
{
	int lRows = left -> rows;
	int lColl = left -> coll;
	int rRows = right -> rows;
	int rColl = right -> coll;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++) {
			(result -> mat)[i * rColl + j] = matrixMultiplication (left, right, i, j);
		}
	}
}

void addMultiplyMatrices (Matrix *left, Matrix *right, Matrix *result)
{
	int lRows = left -> rows;
	int lColl = left -> coll;
	int rRows = right -> rows;
	int rColl = right -> coll;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++) {
			(result -> mat)[i * rColl + j] += matrixMultiplication (left, right, i, j);
		}
	}
}

void deleteMatrix (Matrix *mat)
{
	if (mat == NULL)
		return;

	free (mat -> mat);
	free (mat);
	mat = NULL;
}

Layer* createLayer (int rows, int coll)
{
	Layer *layer = malloc (sizeof (Layer));

	layer -> weights = createMatrix (rows, coll);
	layer -> weightsT = createMatrix (coll, rows);
	layer -> biases = createMatrix (1, coll);
	layer -> output = createMatrix (1, coll);
	layer -> outputT = createMatrix (coll, 1);
	layer -> weightGradient = createMatrix (rows, coll);
	layer -> biasGradient = createMatrix (1, coll);
	layer -> inputGradient = createMatrix (1, coll);
	layer -> costGradient = createMatrix (1, coll);

	return layer;
}

void randomizeLayer (Layer *lay, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound)
{
	randomMatrix (lay -> weights, lowerWeightBound, upperWeightBound);
	transposeMatrix (lay -> weights, lay -> weightsT);
	randomMatrix (lay -> biases, lowerBiasBound, upperBiasBound);
}

void printLayer (Layer *lay)
{
	printMatrix (lay -> weights);
	printMatrix (lay -> biases);
}

void deleteLayer (Layer *lay)
{
	deleteMatrix (lay -> weights);
	deleteMatrix (lay -> weightsT);
	deleteMatrix (lay -> biases);
	deleteMatrix (lay -> output);
	deleteMatrix (lay -> outputT);
	deleteMatrix (lay -> weightGradient);
	deleteMatrix (lay -> biasGradient);
	deleteMatrix (lay -> inputGradient);
	deleteMatrix (lay -> costGradient);
	free (lay);
	lay = NULL;
}

NeuralNet* createNeuralNet (int numOfLayers, int *neurons)
{
	NeuralNet *net = malloc (sizeof (NeuralNet));
	net -> layers = malloc (sizeof (Layer*) * (numOfLayers - 1));
	net -> layerActivationFunctions = malloc (sizeof (int) * (numOfLayers - 1));
	net -> neurons = malloc (sizeof (int) * numOfLayers);

	net -> numOfLayers = numOfLayers;

	for (int i = 0; i < numOfLayers; i++)
		(net -> neurons)[i] = neurons[i];

	for (int i = 0; i < numOfLayers - 1; i++) {
		(net -> layers)[i] = createLayer (neurons[i], neurons[i + 1]);
		(net -> layers)[i] -> actFunc = &(activationFunctions[0]);
	}

	net -> costFunction = &(costFunctions[0]);
	net -> netCostFunction = 0;

	return net;
}

void initializeNeuralNet (NeuralNet *net, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound)
{
	for (int i = 0; i < (net -> numOfLayers) - 1; i++)
		randomizeLayer ((net -> layers)[i], lowerWeightBound, upperWeightBound, lowerBiasBound, upperBiasBound);
}

void printNeuralNet (NeuralNet *net)
{
	for (int i = 0; i < (net -> numOfLayers) - 1; i++) {
		printf ("Layer %d:\n\n", i);
		printLayer ((net -> layers)[i]);
	}
}

void printGradient (NeuralNet *net)
{
	int length = net -> numOfLayers - 1;
	Layer **layers = net -> layers;

	for (int i = 0; i < length; i++) {
		printf ("Gradient %d\n\n", i);
		printMatrix (layers[i] -> weightGradient);
		printMatrix (layers[i] -> inputGradient);
	}
}

void setActivationFunctions (NeuralNet *net, int *actFuncs)
{
	for (int i = 0; i < (net -> numOfLayers) - 1; i++) {
		(net -> layerActivationFunctions)[i] = actFuncs[i];
		(net -> layers)[i] -> actFunc = &(activationFunctions[actFuncs[i]]);
	}
}

void setCostFunction (NeuralNet *net, int costFunc)
{
	net -> netCostFunction = costFunc;
	net -> costFunction = &(costFunctions[costFunc]);
}

void setAlpha (NeuralNet *net, double alpha)
{
	net -> alpha = -alpha;
}

void forwardPassThroughLayer (Layer *lay, Matrix *input)
{
	multiplyMatrices (input, lay -> weights, lay -> output);
	addMatrices (lay -> output, lay -> biases);
	lay -> actFunc -> function (lay -> output);
}

void forwardPropagation (NeuralNet *net, Matrix *input)
{
	Layer **layers = net -> layers;
	forwardPassThroughLayer (layers[0], input);

	int numOfLayers = net -> numOfLayers;
	for (int i = 1; i < numOfLayers - 1; i++)
		forwardPassThroughLayer (layers[i], layers[i - 1] -> output);
}

void multiplyGradient (Matrix *inputGradient, Matrix *costGradient)
{
	int length = inputGradient -> coll;

	for (int i = 0; i < length; i++)
		inputGradient -> mat[i] *= costGradient -> mat[i];
}

void backPassThroughLayer (Layer *currLay, Layer *prevLay)
{
	addMatrices (currLay -> biasGradient, currLay -> inputGradient);
	transposeMatrix (prevLay -> output, prevLay -> outputT);
	addMultiplyMatrices (prevLay -> outputT, currLay -> inputGradient, currLay -> weightGradient);

	multiplyMatrices (currLay -> inputGradient, currLay -> weightsT, prevLay -> costGradient);
	prevLay -> actFunc -> derFunction (prevLay -> output, prevLay -> inputGradient);
	multiplyGradient (prevLay -> inputGradient, prevLay -> costGradient);
}

void backPropagation (NeuralNet *net, Matrix *inputT, Matrix *y)
{
	
	Layer **layers = net -> layers;
	int numOfLayers = net -> numOfLayers;

	net -> costFunction -> derFunction (layers[numOfLayers - 2] -> output, y, layers[numOfLayers - 2] -> costGradient);
	layers[numOfLayers - 2] -> actFunc -> derFunction (layers[numOfLayers - 2] -> output, layers[numOfLayers - 2] -> inputGradient);
	multiplyGradient (layers[numOfLayers - 2] -> inputGradient, layers[numOfLayers - 2] -> costGradient);

	for (int i = numOfLayers - 2; i > 0; i--)
		backPassThroughLayer (layers[i], layers[i - 1]);

	addMatrices (net -> layers[0] -> biasGradient, net -> layers[0] -> inputGradient);
	addMultiplyMatrices (inputT, net -> layers[0] -> inputGradient, net -> layers[0] -> weightGradient);
}

void clearGradient (NeuralNet *net)
{
	Layer **layers = net -> layers;
	int numOfLayers = net -> numOfLayers;

	for (int i = 0; i < numOfLayers - 1; i++) {
		resetMatrix (layers[i] -> weightGradient);
		resetMatrix (layers[i] -> biasGradient);
		resetMatrix (layers[i] -> inputGradient);
	}
}

void applyGradient (NeuralNet *net, int batchSize)
{
	Layer **layers = net -> layers;
	int numOfLayers = net -> numOfLayers;
	double scaleFactor = 1.0 / batchSize;

	for (int i = 0; i < numOfLayers - 1; i++) {
		scaleMatrix (layers[i] -> weightGradient, scaleFactor);
		scaleMatrix (layers[i] -> weightGradient, net -> alpha);
		scaleMatrix (layers[i] -> biasGradient, scaleFactor);
		scaleMatrix (layers[i] -> biasGradient, net -> alpha);
	}

	for (int i = 0; i < numOfLayers - 1; i++) {
		addMatrices (layers[i] -> weights, layers[i] -> weightGradient);
		addMatrices (layers[i] -> biases, layers[i] -> biasGradient);
	}
}

void convertInputArraysToMatrices (NeuralNet *net, int numOfExamples, double *dataSet)
{
	Matrix *tmp1, *tmp2;
	Matrix **data = net -> dataSet;
	int inputSize = net -> neurons[0];
	int outputSize = net -> neurons[net -> numOfLayers - 1];

	for (int i = 0; i < numOfExamples; i++) {
		tmp1 = createMatrix (1, inputSize);
		tmp2 = createMatrix (inputSize, 1);

		for (int j = 0; j < inputSize; j++)
			tmp1 -> mat[j] = dataSet[i * (inputSize + outputSize) + j];

		transposeMatrix (tmp1, tmp2);
		data[i * 3] = tmp1;
		data[i * 3 + 1] = tmp2;

		tmp1 = createMatrix (1, outputSize);

		for (int j = 0; j < outputSize; j++)
			tmp1 -> mat[j] = dataSet[i * (inputSize + outputSize) + inputSize + j];

		data[i * 3 + 2] = tmp1;
	}
}

void trainNeuralNet (NeuralNet *net, int numOfBatches, int *batches, double *dataSet)
{
	int numOfExamples = 0;
	for (int i = 0; i < numOfBatches; i++)
		numOfExamples += batches[i];

	net -> dataSet = malloc (sizeof (Matrix*) * numOfExamples * 3);

	convertInputArraysToMatrices (net, numOfExamples, dataSet);

	Matrix **data = net -> dataSet;
	for (int i = 0; i < numOfBatches; i++) {
		for (int j = 0; j < batches[i]; j++) {
			forwardPropagation (net, data[i * 3]);
			backPropagation (net, data[i * 3 + 1], data[i * 3 + 2]);
		}
		applyGradient (net, batches[i]);
		clearGradient (net);
	}

	for (int i = 0; i < numOfExamples * 3; i++)
		deleteMatrix (data[i]);
	free (data);
}

double neuralNetCost (NeuralNet *net, int numOfExamples, double *dataSet)
{
	double sum = 0;
	net -> dataSet = malloc (sizeof (Matrix*) *numOfExamples * 3);

	convertInputArraysToMatrices (net, numOfExamples, dataSet);

	Matrix **data = net -> dataSet;
	for (int i = 0; i < numOfExamples; i++) {
		forwardPropagation (net, data[i * 3]);
//		printMatrix (data[i * 3 + 1]);
		sum += net -> costFunction -> function (data[i * 3 + 1], data[i * 3 + 2]);
	}

	return sum / numOfExamples;
}
