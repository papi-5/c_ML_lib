/*
#ifndef MACHL_H
#define MACHL_H

struct matrix {
	double *mat;
	int row;
	int col;
};

typedef struct matrix Matrix;

struct mcl_activation {
	void (*function) (Matrix*);
	void (*function_d) (Matrix*, Matrix*);
};

typedef struct mcl_activation mcl_activation;

struct mcl_cost {
	double (*function) (Matrix*, Matrix*);
	void (*function_d) (Matrix*, Matrix*, Matrix*);
};

typedef struct mcl_cost mcl_cost;

struct mcl_layer {
	Matrix *weights;
	Matrix *weights_t;
	Matrix *biases;
	Matrix *output;
	Matrix *output_t;
	Matrix *weightGradient;
	Matrix *biasGradient;
	Matrix *inputGradient;
	Matrix *costGradient;
	mcl_activation *actFunc;
};

typedef struct mcl_layer mcl_layer;

struct neuralNet {
	mcl_layer **layers;
	int *neurons;
	int *layerActivationFunctions;
	int num_layers;
	mcl_cost *mcl_cost;
	int cost_id;
	Matrix **dataSet;
	double alpha;
};

typedef struct neuralNet NeuralNet;

void mcl_sigmoid (Matrix *mat);

void mcl_sigmoid_d (Matrix *mat, Matrix *res);

void mcl_tanh (Matrix *mat);

void mcl_tanh_d (Matrix *mat, Matrix *res);

void mcl_relu (Matrix *mat);

void mcl_relu_d (Matrix *mat, Matrix *res);

void mcl_softmax (Matrix *mat);

void mcl_softmax_d (Matrix *mat, Matrix *res);

double mcl_mse (Matrix *mat, Matrix *y);

void mcl_mse_d (Matrix *mat, Matrix *y, Matrix *res);

double mcl_cross_entropy (Matrix *mat, Matrix *y);

void mcl_cross_entropy_d (Matrix *mat, Matrix *y, Matrix *res);

void printMatrix (Matrix *mat);

Matrix* createMatrix (int row, int col);

void randomMatrix (Matrix *mat, double lowerBound, double upperBound);

Matrix* createRandomMatrix (int row, int col, double lowerBound, double upperBound);

void resetMatrix (Matrix *mat);

void transposeMatrix (Matrix *mat, Matrix *matT);

void scaleMatrix (Matrix *mat, double factor);

void addMatrices (Matrix *matA, Matrix *matB);

double matrixMultiplication (Matrix *left, Matrix *right, int row, int col);

void multiplyMatrices (Matrix *left, Matrix *right, Matrix *result);

void addMultiplyMatrices (Matrix *left, Matrix *right, Matrix *result);

void deleteMatrix (Matrix *mat);

mcl_layer* mcl_layer_create (int row, int col);

void randomizeLayer (mcl_layer *lay, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound);

void mcl_layer_print (mcl_layer *lay);

void mcl_layer_delete (mcl_layer *lay);

NeuralNet* createNeuralNet (int num_layers, int *neurons);

void initializeNeuralNet (NeuralNet *net, double lowerWeightBound, double upperWeightBound, double lowerBiasBound, double upperBiasBound);

void printNeuralNet (NeuralNet *net);

void mcl_network_print_grad (NeuralNet *net);

void setActivationFunctions (NeuralNet *net, int *act_funcs);

void mcl_network_set_cost (NeuralNet *net, int cost_func);

void setAlpha (NeuralNet *net, double alpha);

void forwardPassThroughLayer (mcl_layer *lay, Matrix *input);

void forwardPropagation (NeuralNet *net, Matrix *input);

void multiplyGradient (Matrix *inputGradient, Matrix *costGradient);

void backPassThroughLayer (mcl_layer *currLay, mcl_layer *prevLay);

void backPropagation (NeuralNet *net, Matrix *inputT, Matrix *y);

void clearGradient (NeuralNet *net);

void applyGradient (NeuralNet *net, int batchSize);

void convertInputArraysToMatrices (NeuralNet *net, int numOfExamples, double *dataSet);

void trainNeuralNet (NeuralNet *net, int numOfBatches, int *batches, double *dataSet);

double neuralNetCost (NeuralNet *net, int numOfExamples, double *dataSet);

#endif
*/