/*
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "machl.h"

mcl_activation activation_functions[] = {{mcl_sigmoid, mcl_sigmoid_d},
					{mcl_tanh, mcl_tanh_d},
					{mcl_relu, mcl_relu_d},
					{mcl_softmax, mcl_softmax_d}};

mcl_cost cost_functions[] = {{mcl_mse, mcl_mse_d},
				{mcl_cross_entropy, mcl_cross_entropy_d}};

void mcl_sigmoid (Matrix *mat)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++)
		(mat -> mat)[i] = 1.0 / (1.0 + exp ((mat -> mat)[i] * -1));
}

void mcl_sigmoid_d (Matrix *mat, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		(res -> mat)[i] = x * (1.0 - x);
	}
}

void mcl_tanh (Matrix *mat)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		double tmp1 = exp (x);
		double tmp2 = exp (-x);
		(mat -> mat)[i] = (tmp1 - tmp2) / (tmp1 + tmp2);
	}
}

void mcl_tanh_d (Matrix *mat, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = 1.0 - pow ((mat -> mat)[i], 2.0);
}

void mcl_relu (Matrix *mat)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++) {
		if ((mat -> mat)[i] <= 0)
			(mat -> mat)[i] = 0;
	}
}

void mcl_relu_d (Matrix *mat, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++) {
		if ((mat -> mat)[i] > 0)
			(res -> mat)[i] = 1;
		else
			(res -> mat)[i] = 0;
	}
}

void mcl_softmax (Matrix *mat)
{
	int length = mat -> col * mat -> row;
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

void mcl_softmax_d (Matrix *mat, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++) {
		double x = (mat -> mat)[i];
		(res -> mat)[i] = x * (1.0 - x);
	}
}

double mcl_mse (Matrix *mat, Matrix *y)
{
	int length = mat -> col * mat -> row;
	double res = 0;

	for (int i = 0; i < length; i++) {
		double tmp = (mat -> mat)[i] - (y -> mat)[i];
		res +=	pow (tmp, 2.0);
	}

	res /= length;

	return res;
}

void mcl_mse_d (Matrix *mat, Matrix *y, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = 2.0 * ((mat -> mat)[i] - (y -> mat)[i]);
}

double mcl_cross_entropy (Matrix *mat, Matrix *y)
{
	int length = mat -> col * mat -> row;
	double res = 0;

	for (int i = 0; i < length; i++)
		res += (y -> mat)[i] * log ((mat -> mat)[i]);

	res *= -1.0;

	return res;
}

void mcl_cross_entropy_d (Matrix *mat, Matrix *y, Matrix *res)
{
	int length = mat -> col * mat -> row;

	for (int i = 0; i < length; i++)
		(res -> mat)[i] = -1.0 * ((y -> mat)[i] / (mat -> mat)[i]);
}



void printMatrix (Matrix *mat)
{
	int row = mat -> row;
	int col = mat -> col;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf ("%lf ", (mat -> mat)[i * col + j]);
		}
		printf("\n");
	}
	printf("\n");
}

Matrix* createMatrix (int row, int col)
{
	Matrix *mat = malloc (sizeof (Matrix));

	mat -> row = row;
	mat -> col = col;
	mat -> mat = calloc (row * col, sizeof (double));

	return mat;
}

void randomMatrix (Matrix *mat, double lowerBound, double upperBound)
{
	int row = mat -> row;
	int col = mat -> col;

	srand (time (NULL));

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			(mat -> mat)[i * col + j] = ((double)rand() / RAND_MAX) * (upperBound - lowerBound) + lowerBound;
		}
	}
}

Matrix* createRandomMatrix (int row, int col, double lowerBound, double upperBound)
{
	Matrix *mat = createMatrix (row, col);
	randomMatrix (mat, lowerBound, upperBound);

	return mat;
}

void resetMatrix (Matrix *mat)
{
	int length = mat -> row * mat -> col;

	for (int i = 0; i < length; i++)
		mat -> mat[i] = 0;
}

void transposeMatrix (Matrix *mat, Matrix *matT)
{
	int row = mat -> row;
	int col = mat -> col;

	if (row != matT -> col
		|| col != matT -> row)
		return;

	for (int i = 0; i < row * col; i++) {
		(matT -> mat)[i] = (mat -> mat)[(i % row) * col + i / row];
	}
}

void scaleMatrix (Matrix *mat, double factor)
{
	int length = mat -> row * mat -> col;

	for (int i = 0; i < length; i++)
		mat -> mat[i] *= factor;
}

void addMatrices (Matrix *matA, Matrix *matB)
{
	int length = matA -> row * matA -> col;

	for (int i = 0; i < length; i++)
		(matA -> mat)[i] += (matB -> mat)[i];
}

double matrixMultiplication (Matrix *left, Matrix *right, int row, int col)
{
	int length = left -> col;
	int lColl = left -> col;
	int rColl = right -> col;
	double *lMat = left -> mat;
	double *rMat = right -> mat;
	double sum = 0;

	for (int i = 0; i < length; i++) {
		sum += lMat[row * lColl + i] * rMat[i * rColl + col];
	}

	return sum;
}

void multiplyMatrices (Matrix *left, Matrix *right, Matrix *result)
{
	int lRows = left -> row;
	int lColl = left -> col;
	int rRows = right -> row;
	int rColl = right -> col;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++) {
			(result -> mat)[i * rColl + j] = matrixMultiplication (left, right, i, j);
		}
	}
}

void addMultiplyMatrices (Matrix *left, Matrix *right, Matrix *result)
{
	int lRows = left -> row;
	int lColl = left -> col;
	int rRows = right -> row;
	int rColl = right -> col;

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

mcl_layer* mcl_layer_create (int row, int col)
{
	mcl_layer *mcl_layer = malloc (sizeof (mcl_layer));

	mcl_layer -> weights = createMatrix (row, col);
	mcl_layer -> weights_t = createMatrix (col, row);
	mcl_layer -> biases = createMatrix (1, col);
	mcl_layer -> output = createMatrix (1, col);
	mcl_layer -> output_t = createMatrix (col, 1);
	mcl_layer -> weightGradient = createMatrix (row, col);
	mcl_layer -> biasGradient = createMatrix (1, col);
	mcl_layer -> inputGradient = createMatrix (1, col);
	mcl_layer -> costGradient = createMatrix (1, col);

	return mcl_layer;
}

void randomizeLayer (mcl_layer *lay, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound)
{
	randomMatrix (lay -> weights, lowerWeightBound, upperWeightBound);
	transposeMatrix (lay -> weights, lay -> weights_t);
	randomMatrix (lay -> biases, lowerBiasBound, upperBiasBound);
}

void mcl_layer_print (mcl_layer *lay)
{
	printMatrix (lay -> weights);
	printMatrix (lay -> biases);
}

void mcl_layer_delete (mcl_layer *lay)
{
	deleteMatrix (lay -> weights);
	deleteMatrix (lay -> weights_t);
	deleteMatrix (lay -> biases);
	deleteMatrix (lay -> output);
	deleteMatrix (lay -> output_t);
	deleteMatrix (lay -> weightGradient);
	deleteMatrix (lay -> biasGradient);
	deleteMatrix (lay -> inputGradient);
	deleteMatrix (lay -> costGradient);
	free (lay);
	lay = NULL;
}

NeuralNet* createNeuralNet (int num_layers, int *neurons)
{
	NeuralNet *net = malloc (sizeof (NeuralNet));
	net -> layers = malloc (sizeof (mcl_layer*) * (num_layers - 1));
	net -> layerActivationFunctions = malloc (sizeof (int) * (num_layers - 1));
	net -> neurons = malloc (sizeof (int) * num_layers);

	net -> num_layers = num_layers;

	for (int i = 0; i < num_layers; i++)
		(net -> neurons)[i] = neurons[i];

	for (int i = 0; i < num_layers - 1; i++) {
		(net -> layers)[i] = mcl_layer_create (neurons[i], neurons[i + 1]);
		(net -> layers)[i] -> actFunc = &(activation_functions[0]);
	}

	net -> mcl_cost = &(cost_functions[0]);
	net -> cost_id = 0;

	return net;
}

void initializeNeuralNet (NeuralNet *net, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound)
{
	for (int i = 0; i < (net -> num_layers) - 1; i++)
		randomizeLayer ((net -> layers)[i], lowerWeightBound, upperWeightBound, lowerBiasBound, upperBiasBound);
}

void printNeuralNet (NeuralNet *net)
{
	for (int i = 0; i < (net -> num_layers) - 1; i++) {
		printf ("mcl_layer %d:\n\n", i);
		mcl_layer_print ((net -> layers)[i]);
	}
}

void mcl_network_print_grad (NeuralNet *net)
{
	int length = net -> num_layers - 1;
	mcl_layer **layers = net -> layers;

	for (int i = 0; i < length; i++) {
		printf ("Gradient %d\n\n", i);
		printMatrix (layers[i] -> weightGradient);
		printMatrix (layers[i] -> inputGradient);
	}
}

void setActivationFunctions (NeuralNet *net, int *act_funcs)
{
	for (int i = 0; i < (net -> num_layers) - 1; i++) {
		(net -> layerActivationFunctions)[i] = act_funcs[i];
		(net -> layers)[i] -> actFunc = &(activation_functions[act_funcs[i]]);
	}
}

void mcl_network_set_cost (NeuralNet *net, int cost_func)
{
	net -> cost_id = cost_func;
	net -> mcl_cost = &(cost_functions[cost_func]);
}

void setAlpha (NeuralNet *net, double alpha)
{
	net -> alpha = -alpha;
}

void forwardPassThroughLayer (mcl_layer *lay, Matrix *input)
{
	multiplyMatrices (input, lay -> weights, lay -> output);
	addMatrices (lay -> output, lay -> biases);
	lay -> actFunc -> function (lay -> output);
}

void forwardPropagation (NeuralNet *net, Matrix *input)
{
	mcl_layer **layers = net -> layers;
	forwardPassThroughLayer (layers[0], input);

	int num_layers = net -> num_layers;
	for (int i = 1; i < num_layers - 1; i++)
		forwardPassThroughLayer (layers[i], layers[i - 1] -> output);
}

void multiplyGradient (Matrix *inputGradient, Matrix *costGradient)
{
	int length = inputGradient -> col;

	for (int i = 0; i < length; i++)
		inputGradient -> mat[i] *= costGradient -> mat[i];
}

void backPassThroughLayer (mcl_layer *currLay, mcl_layer *prevLay)
{
	addMatrices (currLay -> biasGradient, currLay -> inputGradient);
	transposeMatrix (prevLay -> output, prevLay -> output_t);
	addMultiplyMatrices (prevLay -> output_t, currLay -> inputGradient, currLay -> weightGradient);

	multiplyMatrices (currLay -> inputGradient, currLay -> weights_t, prevLay -> costGradient);
	prevLay -> actFunc -> function_d (prevLay -> output, prevLay -> inputGradient);
	multiplyGradient (prevLay -> inputGradient, prevLay -> costGradient);
}

void backPropagation (NeuralNet *net, Matrix *inputT, Matrix *y)
{
	
	mcl_layer **layers = net -> layers;
	int num_layers = net -> num_layers;

	net -> mcl_cost -> function_d (layers[num_layers - 2] -> output, y, layers[num_layers - 2] -> costGradient);
	layers[num_layers - 2] -> actFunc -> function_d (layers[num_layers - 2] -> output, layers[num_layers - 2] -> inputGradient);
	multiplyGradient (layers[num_layers - 2] -> inputGradient, layers[num_layers - 2] -> costGradient);

	for (int i = num_layers - 2; i > 0; i--)
		backPassThroughLayer (layers[i], layers[i - 1]);

	addMatrices (net -> layers[0] -> biasGradient, net -> layers[0] -> inputGradient);
	addMultiplyMatrices (inputT, net -> layers[0] -> inputGradient, net -> layers[0] -> weightGradient);
}

void clearGradient (NeuralNet *net)
{
	mcl_layer **layers = net -> layers;
	int num_layers = net -> num_layers;

	for (int i = 0; i < num_layers - 1; i++) {
		resetMatrix (layers[i] -> weightGradient);
		resetMatrix (layers[i] -> biasGradient);
		resetMatrix (layers[i] -> inputGradient);
	}
}

void applyGradient (NeuralNet *net, int batchSize)
{
	mcl_layer **layers = net -> layers;
	int num_layers = net -> num_layers;
	double scaleFactor = 1.0 / batchSize;

	for (int i = 0; i < num_layers - 1; i++) {
		scaleMatrix (layers[i] -> weightGradient, scaleFactor);
		scaleMatrix (layers[i] -> weightGradient, net -> alpha);
		scaleMatrix (layers[i] -> biasGradient, scaleFactor);
		scaleMatrix (layers[i] -> biasGradient, net -> alpha);
	}

	for (int i = 0; i < num_layers - 1; i++) {
		addMatrices (layers[i] -> weights, layers[i] -> weightGradient);
		addMatrices (layers[i] -> biases, layers[i] -> biasGradient);
	}
}

void convertInputArraysToMatrices (NeuralNet *net, int numOfExamples, double *dataSet)
{
	Matrix *tmp1, *tmp2;
	Matrix **data = net -> dataSet;
	int inputSize = net -> neurons[0];
	int outputSize = net -> neurons[net -> num_layers - 1];

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
		sum += net -> mcl_cost -> function (data[i * 3 + 1], data[i * 3 + 2]);
	}

	return sum / numOfExamples;
}
*/