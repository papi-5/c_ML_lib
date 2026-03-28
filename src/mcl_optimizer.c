#include <stdlib.h>
#include <math.h>
#include "mcl_optimizer.h"
#include "mcl_tensor.h"
#include "mcl_function.h"
#include "mcl_network.h"
#include "mcl_dataset.h"

extern mcl_cost cost_functions[];

mcl_optimizer* mcl_optimizer_create ()
{
    mcl_optimizer *opt = malloc (sizeof (mcl_optimizer));

    opt -> net = NULL;
    opt -> cost_id = MCL_MSE;
    opt -> cost = &(cost_functions[0]);
    opt -> dropout = 0;
    opt -> beta1 = 0.9;
    opt -> beta2 = 0.999;
    opt -> epsilon = 1e-8;
    opt -> timestep = 1;
    opt -> m = NULL;
    opt -> v = NULL;

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

void mcl_optimizer_set_cost (mcl_optimizer *opt, mcl_cost_type cost_id)
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

static void train_batch_sgd (mcl_optimizer *opt, mcl_tensor **batch, int batch_size)
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
            train_batch_sgd (opt, &(dataset_train[j * 2]), b_size);
        }
    }
}

static void initialize_adam (mcl_optimizer *opt)
{
    mcl_network *net = opt -> net;

    if ((net == NULL) || (opt -> m != NULL))
        return;

    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;

    opt -> m = malloc (sizeof (mcl_tensor*) * length * 2);
    opt -> v = malloc (sizeof (mcl_tensor*) * length * 2);

    for (int i = 0; i < length; i++) {
        int row = layers[i] -> weights -> row;
        int col = layers[i] -> weights -> col;
        
        opt -> m[i * 2] = mcl_tensor_create (row, col);
        opt -> m[i * 2 + 1] = mcl_tensor_create (row, 1);
        opt -> v[i * 2] = mcl_tensor_create (row, col);
        opt -> v[i * 2 + 1] = mcl_tensor_create (row, 1);
    }
}

static void calculate_grad_adam (mcl_network *net, mcl_tensor **m, mcl_tensor **v,
                                 float beta1, float beta2, float epsilon, int t,
                                 float learn_rate, int batch_size)
{
    int length = net -> num_layers - 1;
    mcl_layer **layers = net -> layers;
    float inv_bc1 = 1.0f / (1 - pow (beta1, t));
    float inv_bc2 = 1.0f / (1 - pow (beta2, t));

    for (int i = 0; i < length; i++) {
        mcl_tensor *wg = layers[i] -> weight_grad;
        mcl_tensor *bg = layers[i] -> bias_grad;
        mcl_tensor *wm = m[i * 2];
        mcl_tensor *bm = m[i * 2 + 1];
        mcl_tensor *wv = v[i * 2];
        mcl_tensor *bv = v[i * 2 + 1];
        int w_length = wg -> row * wg -> col;
        int b_length = bg -> row * bg -> col;

        for (int j = 0; j < w_length; j++) {
            wg -> ten[j] /= batch_size;
            wm -> ten[j] = beta1 * wm -> ten[j] + (1 - beta1) * wg -> ten[j];
            wv -> ten[j] = beta2 * wv -> ten[j] + (1 - beta2) * pow (wg -> ten[j], 2);
            wg -> ten[j] = -learn_rate * (wm -> ten[j] * inv_bc1) / (sqrt (wv -> ten[j] * inv_bc2) + epsilon); 
        }
        for (int j = 0; j < b_length; j++) {
            bg -> ten[j] /= batch_size;
            bm -> ten[j] = beta1 * bm -> ten[j] + (1 - beta1) * bg -> ten[j];
            bv -> ten[j] = beta2 * bv -> ten[j] + (1 - beta2) * pow (bg -> ten[j], 2);
            bg -> ten[j] = -learn_rate * (bm -> ten[j] * inv_bc1) / (sqrt (bv -> ten[j] * inv_bc2) + epsilon); 
        }
    }
}

static void train_batch_adam (mcl_optimizer *opt, mcl_tensor **batch, int batch_size)
{
    mcl_network *net = opt -> net;
    mcl_tensor *output = net -> layers[net -> num_layers - 2] -> output;
    mcl_tensor *output_grad = net -> layers[net -> num_layers - 2] -> output_grad;
    mcl_cost *cost = opt -> cost;
    int t = opt -> timestep;
    float beta1 = opt -> beta1;
    float beta2 = opt -> beta2;
    float epsilon = opt -> epsilon;
    float learn_rate = opt -> learn_rate;
    float dropout = opt -> dropout;

    mcl_network_reset_grad (net);

    for (int i = 0; i < batch_size; i++) {
        mcl_network_forward_train (net, batch[i * 2], dropout);
        cost -> function_d (output, batch[i * 2 + 1], output_grad);
        mcl_network_backward (net, batch[i * 2]);
    }

    calculate_grad_adam (net, opt -> m, opt -> v, 
                         beta1, beta2, epsilon, t,
                         learn_rate, batch_size);
    mcl_network_apply_grad (net);

    opt -> timestep++;
}

void mcl_optimizer_train_adam (mcl_optimizer *opt, int batch_size, int num_epochs)
{
    initialize_adam (opt);

    int train_size = opt -> data -> train_size;
    mcl_tensor **dataset_train = opt -> data -> train;

    for (int i = 0; i < num_epochs; i++) {
        mcl_dataset_shuffle (dataset_train, train_size);
        
        for (int j = 0; j < train_size; j += batch_size) {
            int b_size = train_size - j < batch_size ? train_size - j : batch_size;
            train_batch_adam (opt, &(dataset_train[j * 2]), b_size);
        }
    }
}