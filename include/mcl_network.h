#ifndef MCL_NET_H
#define MCL_NET_H

#include "mcl_tensor.h"
#include "mcl_function.h"
#include "mcl_layer.h"

typedef struct mcl_network {
    mcl_layer **layers;
    int *neurons;
    int *act_ids;
    int num_layers;
    mcl_cost *cost;
    int cost_id;
    mcl_tensor **dataset;
    float learn_rate;
    float dropout;
} mcl_network;

mcl_network* mcl_network_create (int num_layers, int *neurons);

size_t mcl_network_size (mcl_network *net);

void mcl_network_init_xavier_uniform (mcl_network *net);

void mcl_network_init_xavier_normal (mcl_network *net);

void mcl_network_init_kaiming (mcl_network *net);

void mcl_network_print (mcl_network *net);

void mcl_network_print_grad (mcl_network *net);

void mcl_network_set_activations (mcl_network *net, int *act_funcs);

void mcl_network_set_cost (mcl_network *net, int cost_func);

void mcl_network_set_learn_rate (mcl_network *net, float learn_rate);

void mcl_network_set_dropout (mcl_network *net, float dropout);

void mcl_network_delete (mcl_network *net);

#endif