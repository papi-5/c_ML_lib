#ifndef MCL_FUNC_H
#define MCL_FUNC_H

#include "mcl_tensor.h"

typedef struct mcl_activation {
	void (*function) (mcl_tensor*);
	void (*function_d) (mcl_tensor*, mcl_tensor*);
} mcl_activation;

typedef struct mcl_cost {
	float (*function) (mcl_tensor*, mcl_tensor*);
	void (*function_d) (mcl_tensor*, mcl_tensor*, mcl_tensor*);
} mcl_cost;

void mcl_sigmoid (mcl_tensor *ten);

void mcl_sigmoid_d (mcl_tensor *ten, mcl_tensor *res);

void mcl_tanh (mcl_tensor *ten);

void mcl_tanh_d (mcl_tensor *ten, mcl_tensor *res);

void mcl_relu (mcl_tensor *ten);

void mcl_relu_d (mcl_tensor *ten, mcl_tensor *res);

void mcl_softmax (mcl_tensor *ten);

void mcl_softmax_d (mcl_tensor *ten, mcl_tensor *res);

float mcl_mse (mcl_tensor *ten, mcl_tensor *y);

void mcl_mse_d (mcl_tensor *ten, mcl_tensor *y, mcl_tensor *res);

float mcl_cross_entropy (mcl_tensor *ten, mcl_tensor *y);

void mcl_cross_entropy_d (mcl_tensor *ten, mcl_tensor *y, mcl_tensor *res);

#endif
