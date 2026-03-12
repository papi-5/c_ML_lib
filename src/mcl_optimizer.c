#include <stdlib.h>
#include "mcl_optimizer.h"
#include "mcl_tensor.h"
#include "mcl_function.h"
#include "mcl_network.h"
#include "mcl_dataset.h"

extern mcl_cost cost_functions[];

mcl_optimizer* mcl_optimizer_create ()
{
    mcl_optimizer *opt = malloc (sizeof (mcl_optimizer));

    opt -> cost_id = 0;
    opt -> cost = &(cost_functions[0]);
    opt -> dropout = 0;
    opt -> beta1 = 0.9;
    opt -> beta2 = 0.999;
    opt -> epsilon = 1e-8;

    return opt;
}

void mcl_optimizer_set_dataset (mcl_optimizer *opt, mcl_dataset *data)
{
    opt -> data = data;
}

void mcl_optimizer_set_network (mcl_optimizer *opt, mcl_network *net)
{
    opt -> net = net;
}

void mcl_optimizer_set_cost (mcl_optimizer *opt, int cost_id)
{
    opt -> cost = &(cost_functions[cost_id]);
    opt -> cost_id = cost_id;
}

void mcl_optimizer_set_learn_rate (mcl_optimizer *opt, float learn_rate)
{
    opt -> learn_rate = learn_rate;
}

void mcl_optimizer_set_dropout (mcl_optimizer *opt, float dropout)
{
    opt -> dropout = dropout;
}

void mcl_optimizer_set_beta1 (mcl_optimizer *opt, float beta1)
{
    opt -> beta1 = beta1;
}

void mcl_optimizer_set_beta2 (mcl_optimizer *opt, float beta2)
{
    opt -> beta2 = beta2;
}

void mcl_optimizer_set_epsilon (mcl_optimizer *opt, float epsilon)
{
    opt -> epsilon = epsilon;
}

float mcl_optimizer_test_train (mcl_optimizer *opt, int batch_size, float *accuracy)
{
    int train_size = opt -> data -> train_size;
    mcl_tensor **dataset_train = opt -> data -> train;
    batch_size = train_size < batch_size ? train_size : batch_size;
    mcl_network *net = opt -> net;
    mcl_tensor *output = net -> layers[net -> num_layers - 2] -> output;
    mcl_cost *cost = opt -> cost;
    float loss = 0;
    int correct = 0;

    mcl_dataset_shuffle (dataset_train, train_size);

    for (int i = 0; i < batch_size; i++) {
        mcl_network_forward_test (net, dataset_train[i * 2]);
        loss += cost -> function (output, dataset_train[i * 2 + 1]);
        if (mcl_tensor_argmax (output) == mcl_tensor_argmax (dataset_train[i * 2 + 1])) {
            correct++;
        }
    }

    loss /= batch_size;
    *accuracy = (float)correct / batch_size;

    return loss;
}

float mcl_optimizer_test (mcl_optimizer *opt, int batch_size, float *accuracy)
{
    int test_size = opt -> data -> test_size;
    mcl_tensor **dataset_test = opt -> data -> test;
    batch_size = test_size < batch_size ? test_size : batch_size;
    mcl_network *net = opt -> net;
    mcl_tensor *output = net -> layers[net -> num_layers - 2] -> output;
    mcl_cost *cost = opt -> cost;
    float loss = 0;
    int correct = 0;

    mcl_dataset_shuffle (dataset_test, test_size);

    for (int i = 0; i < batch_size; i++) {
        mcl_network_forward_test (net, dataset_test[i * 2]);
        loss += cost -> function (output, dataset_test[i * 2 + 1]);
        if (mcl_tensor_argmax (output) == mcl_tensor_argmax (dataset_test[i * 2 + 1])) {
            correct++;
        }
    }

    loss /= batch_size;
    *accuracy = (float)correct / batch_size;

    return loss;
}

static void train_batch (mcl_optimizer *opt, mcl_tensor **batch, int batch_size)
{
    mcl_network *net = opt -> net;
    mcl_tensor *output = net -> layers[net -> num_layers - 2] -> output;
    mcl_tensor *output_grad = net -> layers[net -> num_layers - 2] -> output_grad;
    mcl_cost *cost = opt -> cost;
    float learn_rate = opt -> learn_rate;
    float dropout = opt -> dropout;

    mcl_network_reset_grad (net);

    for (int i = 0; i < batch_size; i++) {
        mcl_network_forward_train (net, batch[i * 2], dropout);
        cost -> function_d (output, batch[i * 2 + 1], output_grad);
        mcl_network_backward (net, batch[i * 2]);
    }

    mcl_network_scale_grad (net, -learn_rate / batch_size);
    mcl_network_apply_grad (net);
}

void mcl_optimizer_train_sgd (mcl_optimizer *opt, int batch_size, int num_epochs)
{
    int train_size = opt -> data -> train_size;
    mcl_tensor **dataset_train = opt -> data -> train;

    for (int i = 0; i < num_epochs; i++) {
        mcl_dataset_shuffle (dataset_train, train_size);
        
        for (int j = 0; j < train_size; j += batch_size) {
            int b_size = train_size - j < batch_size ? train_size - j : batch_size;
            train_batch (opt, &(dataset_train[j]), b_size);
        }
    }
}