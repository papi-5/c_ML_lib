#ifndef MCL_OPT_H
#define MCL_OPT_H

#include "mcl_tensor.h"
#include "mcl_function.h"
#include "mcl_network.h"
#include "mcl_dataset.h"

typedef struct mcl_optimizer {
    mcl_network *net;
    mcl_dataset *data;
    mcl_cost *cost;
    int cost_id;
    float learn_rate;
    float dropout;
    float beta1;
    float beta2;
    float epsilon;
    int timestep;
    mcl_tensor **m;
    mcl_tensor **v;
} mcl_optimizer;

mcl_optimizer* mcl_optimizer_create ();

void mcl_optimizer_set_dataset (mcl_optimizer *opt, mcl_dataset *data);

void mcl_optimizer_set_network (mcl_optimizer *opt, mcl_network *net);

void mcl_optimizer_set_cost (mcl_optimizer *opt, int cost_id);

#endif