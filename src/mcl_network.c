#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mcl_tensor.h"
#include "mcl_function.h"
#include "mcl_layer.h"
#include "mcl_network.h"

extern mcl_activation activation_functions[];
extern mcl_cost cost_functions[];

mcl_network* mcl_network_create (int *neurons, int num_layers)
{
    mcl_network *net = malloc (sizeof (mcl_network));
    net -> layers = malloc (sizeof (mcl_layer*) * (num_layers - 1));
    net -> act_ids = malloc (sizeof (int) * (num_layers - 1));
    net -> neurons = malloc (sizeof (int) * num_layers);

    net -> num_layers = num_layers;

    for (int i = 0; i < num_layers; i++)
        net -> neurons[i] = neurons[i];

    for (int i = 0; i < num_layers - 1; i++) {
        net -> layers[i] = mcl_layer_create(neurons[i + 1], neurons[i]);
        net -> layers[i] -> activation = &(activation_functions[0]);
    }

    net -> cost = &(cost_functions[0]);
    net -> cost_id = 0;

    return net;
}

size_t mcl_network_size (mcl_network *net)
{
    size_t size = 0;
    int length = net -> num_layers;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length - 1; i++)
        size += mcl_layer_size(layers[i]);

    size += sizeof (int) * length;
    size += sizeof (int) * (length - 1);
    size += sizeof (mcl_network);

    return size;
}

void mcl_network_init_xavier_uniform (mcl_network *net)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float x = layers[i] -> weights -> row;
        x += layers[i] -> weights -> col;
        x = sqrt (6 / x);
        mcl_layer_weights_rand_uniform(layers[i], -x, x);
        mcl_layer_biases_rand_uniform(layers[i], -x, x);
    }
}

void mcl_network_init_xavier_normal (mcl_network *net)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float stddev = layers[i] -> weights -> row;
        stddev += layers[i] -> weights -> col;
        stddev = sqrt (2 / stddev);
        mcl_layer_weights_rand_normal(layers[i], 0, stddev);
        mcl_layer_biases_rand_normal(layers[i], 0, stddev);
    }
}

void mcl_network_init_kaiming (mcl_network *net)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        float stddev = layers[i] -> weights -> row;
        stddev = sqrt (2 / stddev);
        mcl_layer_weights_rand_normal(layers[i], 0, stddev);
        mcl_layer_biases_rand_normal(layers[i], 0, stddev);
    }
}

void mcl_network_print (mcl_network *net)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        printf ("Layer %d:\n\n", i);
        mcl_layer_print (layers[i]);
    }
}

void mcl_network_print_grad (mcl_network *net)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        printf ("Gradient %d:\n\n", i);
        mcl_tensor_print(layers[i] -> weight_grad);
        mcl_tensor_print(layers[i] -> input_grad);
    }
}

void mcl_network_set_activations (mcl_network *net, int *act_funcs)
{
    int length = net -> num_layers - 1;
    int *act_ids = net -> act_ids;
    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length; i++) {
        act_ids[i] = act_funcs[i];
        layers[i] -> activation = &(activation_functions[act_funcs[i]]);
    }
}

void mcl_network_set_cost (mcl_network *net, int cost_func)
{
    net -> cost_id = cost_func;
    net -> cost = &(cost_functions[cost_func]);
}

void mcl_network_set_learn_rate (mcl_network *net, float learn_rate)
{
    net -> learn_rate = -learn_rate;
}

void mcl_network_set_dropout (mcl_network *net, float dropout)
{
    if (dropout < 0 || dropout > 1)
        return;

    net -> dropout = dropout;
}

void mcl_network_forward_train (mcl_network *net, mcl_tensor *input, float dropout)
{
    int length = net -> num_layers - 1;
    mcl_layer ** layers = net -> layers;
    net -> input = input;
    for (int i = 0; i < length - 1; i++) {
        mcl_layer_forward_train (layers[i], input, dropout);
        input = layers[i] -> output;
    }
    mcl_layer_forward_test (layers[length - 1], input);
}

void mcl_network_forward_test (mcl_network *net, mcl_tensor *input)
{
    int length = net -> num_layers - 1;
    mcl_layer ** layers = net -> layers;
    for (int i = 0; i < length; i++) {
        mcl_layer_forward_test (layers[i], input);
        input = layers[i] -> output;
    }
}

void mcl_network_delete (mcl_network *net)
{
    if (net == NULL)
        return;
    
    int length = net -> num_layers;

    for (int i = 0; i < length; i++)
        mcl_layer_delete(net -> layers[i]);
    free (net -> neurons);
    free (net -> act_ids);
    free (net);
    net = NULL;
}