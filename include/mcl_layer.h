#ifndef MCL_LAYER_H
#define MCL_LAYER_H

#include "mcl_tensor.h"
#include "mcl_function.h"

typedef struct mcl_layer {
	mcl_tensor *weights;
	mcl_tensor *biases;
	mcl_tensor *output;
	mcl_tensor *weight_grad;
	mcl_tensor *bias_grad;
	mcl_tensor *delta;
	mcl_tensor *output_grad;
	mcl_activation *activation;
} mcl_layer;

mcl_layer* mcl_layer_create (int row, int col);

size_t mcl_layer_size (mcl_layer *lay);

void mcl_layer_weights_rand_uniform (mcl_layer *lay, float lower, float upper);

void mcl_layer_weights_rand_normal (mcl_layer *lay, float mean, float stddev);

void mcl_layer_biases_rand_uniform (mcl_layer *lay, float lower, float upper);

void mcl_layer_biases_rand_normal (mcl_layer *lay, float mean, float stddev);

void mcl_layer_print (mcl_layer *lay);

void mcl_layer_forward_train (mcl_layer *lay, mcl_tensor *input, float dropout);

void mcl_layer_forward_test (mcl_layer *lay, mcl_tensor *input);

void mcl_layer_backward (mcl_layer *lay, mcl_tensor *input, mcl_tensor *output_grad);

void mcl_layer_delete (mcl_layer *lay);

#endif
