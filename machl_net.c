#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "machl_tensor.h"
#include "machl_function.h"
#include "machl_layer.h"
#include "machl_net.h"

extern ActFunction activationFunctions[];
extern CostFunction costFunctions[];

Net* createNet (int numOfLayers, int *neurons)
{
    Net *net = malloc (sizeof (Net));
    net -> layers = malloc (sizeof (Layer*) * (numOfLayers - 1));
    net -> layerActFunctions = malloc (sizeof (int) * (numOfLayers - 1));
    net -> neurons = malloc (sizeof (int) * numOfLayers);

    net -> numOfLayers = numOfLayers;

    for (int i = 0; i < numOfLayers; i++)
        net -> neurons[i] = neurons[i];

    for (int i = 0; i < numOfLayers - 1; i++) {
        net -> layers[i] = createLayer(neurons[i], neurons[i + 1]);
        net -> layers[i] -> actFunction = &(activationFunctions[0]);
    }

    net -> costFunction = &(costFunctions[0]);
    net -> netCostFunction = 0;

    return net;
}

void uXavier (Net *net)
{
    int length = net -> numOfLayers - 1;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float x = layers[i] -> weights -> rows;
        x += layers[i] -> weights -> coll;
        x = sqrt (6 / x);
        uRandomWeights(layers[i], -x, x);
        uRandomBiases(layers[i], -x, x);
    }
}

void nXavier (Net *net)
{
    int length = net -> numOfLayers - 1;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float stddev = layers[i] -> weights -> rows;
        stddev += layers[i] -> weights -> coll;
        stddev = sqrt (2 / stddev);
        nRandomWeights(layers[i], 0, stddev);
        nRandomBiases(layers[i], 0, stddev);
    }
}

void Kaiming (Net *net)
{
    int length = net -> numOfLayers - 1;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float stddev = layers[i] -> weights -> rows;
        stddev = sqrt (2 / stddev);
        nRandomWeights(layers[i], 0, stddev);
        nRandomBiases(layers[i], 0, stddev);
    }
}

void printNet (Net *net)
{
    int length = net -> numOfLayers - 1;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        printf ("Layer %d:\n\n", i);
        printLayer (layers[i]);
    }
}

void printGradient (Net *net)
{
    int length = net -> numOfLayers - 1;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        printf ("Gradient %d:\n\n", i);
        printTensor(layers[i] -> weightG);
        printTensor(layers[i] -> inputG);
    }
}

void setActFunctions (Net *net, int *actFuncs)
{
    int length = net -> numOfLayers - 1;
    int *layerActFunctions = net -> layerActFunctions;
    Layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        layerActFunctions[i] = actFuncs[i];
        layers[i] -> actFunction = &(activationFunctions[actFuncs[i]]);
    }
}

void setCostFunction (Net *net, int costFunc)
{
    net -> netCostFunction = costFunc;
    net -> costFunction = &(costFunctions[costFunc]);
}

void setLearnRate (Net *net, float learnRate)
{
    net -> learnRate = -learnRate;
}

void setDropout (Net *net, float dropout)
{
    if (dropout < 0 || dropout > 1)
        return;

    net -> dropout = dropout;
}

void deleteNet (Net *net)
{
    if (net == NULL)
        return;
    
    int length = net -> numOfLayers;

    for (int i = 0; i < length; i++)
        deleteLayer(net -> layers[i]);
    free (net -> neurons);
    free (net -> layerActFunctions);
    free (net);
    net = NULL;
}