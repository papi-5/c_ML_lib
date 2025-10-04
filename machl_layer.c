#include <stdlib.h>
#include "machl_tensor.h"
#include "machl_layer.h"

Layer* createLayer (int rows, int coll)
{
	Layer *lay = malloc (sizeof (Layer));

	lay -> weights = createTensor (rows, coll);
	lay -> weightsT = createTensor (coll, rows);
	lay -> biases = createTensor (1, coll);
	lay -> output = createTensor (1, coll);
	lay -> outputT = createTensor (coll, 1);
	lay -> weightG = createTensor (rows, coll);
	lay -> biasG = createTensor (1, coll);
	lay -> inputG = createTensor (1, coll);
	lay -> costG = createTensor (1, coll);

	return lay;
}

void uRandomWeights (Layer *lay, float lowerLimit, float upperLimit)
{
	uRandomTensor (lay -> weights);
	scaleTensor (lay -> weights, upperLimit - lowerLimit);
	addScalar (lay -> weights, lowerLimit);
	transposeTensor (lay -> weights, lay -> weightsT);
}

void nRandomWeights (Layer *lay, float mean, float stddev)
{
	nRandomTensor (lay -> weights);
	scaleTensor (lay -> weights, stddev);
	addScalar (lay -> weights, mean);
	transposeTensor (lay -> weights, lay -> weightsT);
}

void uRandomBiases (Layer *lay, float lowerLimit, float upperLimit)
{
	uRandomTensor (lay -> biases);
	scaleTensor (lay -> biases, upperLimit - lowerLimit);
	addScalar (lay -> biases, lowerLimit);
}

void nRandomBiases (Layer *lay, float mean, float stddev)
{
	nRandomTensor (lay -> biases);
	scaleTensor (lay -> biases, stddev);
	addScalar (lay -> biases, mean);
}

void printLayer (Layer *lay)
{
	printTensor (lay -> weights);
	printTensor (lay -> biases);
}

void deleteLayer (Layer *lay)
{
	if (lay == NULL)
		return;

	deleteTensor (lay -> weights);
	deleteTensor (lay -> weightsT);
	deleteTensor (lay -> biases);
	deleteTensor (lay -> output);
	deleteTensor (lay -> outputT);
	deleteTensor (lay -> weightG);
	deleteTensor (lay -> biasG);
	deleteTensor (lay -> inputG);
	deleteTensor (lay -> costG);
	free (lay);
	lay = NULL;
}
