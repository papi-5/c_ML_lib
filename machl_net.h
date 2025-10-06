#ifndef MACHL_NET_H
#define MACHL_NET_H

#include "machl_tensor.h"
#include "machl_function.h"
#include "machl_layer.h"

struct net {
    Layer **layers;
    int *neurons;
    int *layerActFunctions;
    int numOfLayers;
    CostFunction *costFunction;
    int netCostFunction;
    Tensor **dataset;
    float learnRate;
    float dropout;
};

typedef struct net Net;

Net* createNet (int numOfLayers, int *neurons);

int sizeOfNet (Net *net);

void uXavier (Net *net);

void nXavier (Net *net);

void Kaiming (Net *net);

void printNet (Net *net);

void printGradient (Net *net);

void setActFunctions (Net *net, int *actFuncs);

void setCostFunction (Net *net, int costFunc);

void setLearnRate (Net *net, float learnRate);

void setDropout (Net *net, float dropout);

void deleteNet (Net *net);

#endif