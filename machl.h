#ifndef MACHL_H
#define MACHL_H

struct matrix {
	double *mat;
	int rows;
	int coll;
};

typedef struct matrix Matrix;

struct actFunction {
	void (*function) (Matrix*);
	void (*derFunction) (Matrix*, Matrix*);
};

typedef struct actFunction ActFunction;

struct costFunction {
	double (*function) (Matrix*, Matrix*);
	void (*derFunction) (Matrix*, Matrix*, Matrix*);
};

typedef struct costFunction CostFunction;

struct layer {
	Matrix *weights;
	Matrix *weightsT;
	Matrix *biases;
	Matrix *output;
	Matrix *outputT;
	Matrix *weightGradient;
	Matrix *biasGradient;
	Matrix *inputGradient;
	Matrix *costGradient;
	ActFunction *actFunc;
};

typedef struct layer Layer;

struct neuralNet {
	Layer **layers;
	int *neurons;
	int *layerActivationFunctions;
	int numOfLayers;
	CostFunction *costFunction;
	int netCostFunction;
	Matrix **dataSet;
	double alpha;
};

typedef struct neuralNet NeuralNet;

void sigmoid (Matrix *mat);

void derSigmoid (Matrix *mat, Matrix *res);

void Tanh (Matrix *mat);

void derTanh (Matrix *mat, Matrix *res);

void RElu (Matrix *mat);

void derRElu (Matrix *mat, Matrix *res);

void softmax (Matrix *mat);

void derSoftmax (Matrix *mat, Matrix *res);

double MSE (Matrix *mat, Matrix *y);

void derMSE (Matrix *mat, Matrix *y, Matrix *res);

double crossEntropy (Matrix *mat, Matrix *y);

void derCrossEntropy (Matrix *mat, Matrix *y, Matrix *res);

void printMatrix (Matrix *mat);

Matrix* createMatrix (int rows, int coll);

void randomMatrix (Matrix *mat, double lowerBound, double upperBound);

Matrix* createRandomMatrix (int rows, int coll, double lowerBound, double upperBound);

void resetMatrix (Matrix *mat);

void transposeMatrix (Matrix *mat, Matrix *matT);

void scaleMatrix (Matrix *mat, double factor);

void addMatrices (Matrix *matA, Matrix *matB);

double matrixMultiplication (Matrix *left, Matrix *right, int row, int coll);

void multiplyMatrices (Matrix *left, Matrix *right, Matrix *result);

void addMultiplyMatrices (Matrix *left, Matrix *right, Matrix *result);

void deleteMatrix (Matrix *mat);

Layer* createLayer (int rows, int coll);

void randomizeLayer (Layer *lay, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound);

void printLayer (Layer *lay);

void deleteLayer (Layer *lay);

NeuralNet* createNeuralNet (int numOfLayers, int *neurons);

void initializeNeuralNet (NeuralNet *net, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound);

void printNeuralNet (NeuralNet *net);

void printGradient (NeuralNet *net);

void setActivationFunctions (NeuralNet *net, int *actFuncs);

void setCostFunction (NeuralNet *net, int costFunc);

void setAlpha (NeuralNet *net, double alpha);

void forwardPassThroughLayer (Layer *lay, Matrix *input);

void forwardPropagation (NeuralNet *net, Matrix *input);

void multiplyGradient (Matrix *inputGradient, Matrix *costGradient);

void backPassThroughLayer (Layer *currLay, Layer *prevLay);

void backPropagation (NeuralNet *net, Matrix *inputT, Matrix *y);

void clearGradient (NeuralNet *net);

void applyGradient (NeuralNet *net, int batchSize);

void convertInputArraysToMatrices (NeuralNet *net, int numOfExamples, double *dataSet);

void trainNeuralNet (NeuralNet *net, int numOfBatches, int *batches, double *dataSet);

double neuralNetCost (NeuralNet *net, int numOfExamples, double *dataSet);

#endif
