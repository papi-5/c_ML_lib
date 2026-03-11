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
    opt -> cost = cost_functions[cost_id];
    opt -> dropout = 0;
    opt -> beta1 = 0.9;
    opt -> beta2 = 0.999;
    opt -> epsilon = 1e-8;
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
    opt -> cost = cost_functions[cost_id];
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