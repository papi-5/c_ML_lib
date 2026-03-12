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
} mcl_network;

mcl_network* mcl_network_create (int *neurons, int num_layers);

size_t mcl_network_size (mcl_network *net);

void mcl_network_init_xavier_uniform (mcl_network *net);

void mcl_network_init_xavier_normal (mcl_network *net);

void mcl_network_init_kaiming (mcl_network *net);

void mcl_network_print (mcl_network *net);

void mcl_network_print_grad (mcl_network *net);

void mcl_network_reset_grad (mcl_network *net);

void mcl_network_set_activations (mcl_network *net, int *act_funcs);

void mcl_network_forward_train (mcl_network *net, mcl_tensor *input, float dropout);

void mcl_network_forward_test (mcl_network *net, mcl_tensor *input);

void mcl_network_backward (mcl_network *net, mcl_tensor *input);

void mcl_network_scale_grad (mcl_network *net, float scalar);

void mcl_network_apply_grad (mcl_network *net);

void mcl_network_delete (mcl_network *net);

#endif