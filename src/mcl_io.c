#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mcl_layer.h"
#include "mcl_network.h"
#include "mcl_io.h"

static int digits (int n)
{
    int dig = 0;

    while (n) {
        n /= 10;
        dig++;
    }

    return dig;
}

static int most_digits (int *arr, int n)
{
    int dig = 0;

    for (int i = 0; i < n; i++) {
        int tmp = 0;

        tmp = digits (arr[i]);

        if (tmp > dig)
            dig = tmp;
    }

    return dig;
}

void mcl_network_print_meta (mcl_network *net)
{
    int length = net -> num_layers;
    int width = most_digits(net -> neurons, length);

    printf ("Number of layers: %d\n\n", net -> num_layers);

    printf (" ");
    for (int i = 0; i < length; i++)
        printf ("%*d ", width, i);
    printf ("\n\n");

    printf (" ");
    for (int i = 0; i < length; i++)
        printf ("%*d ", width, net -> neurons[i]);
    printf ("\n\n");

    printf (" ");
    for (int i = 0; i < width + 1; i++)
        printf (" ");
    for (int i = 0; i < length - 1; i++)
        printf ("%*d ", width, net -> act_ids[i]);
    printf ("\n\n");

    printf ("Network cost function: %d\n", net -> cost_id);

    printf ("Learning rate: %f\n", net -> learn_rate);

    printf ("Dropout: %f\n\n", net -> dropout);
}

void mcl_network_export (const char *path, mcl_network *net)
{
    FILE *file = fopen (path, "wb");

    if (!file || !net) {
        printf ("Could not export network.\n");
        return;
    }

    int length = net -> num_layers;
    mcl_layer **layers = net -> layers;

    fwrite (&(net -> num_layers), sizeof (int), 1, file);
    fwrite (&(net -> cost_id), sizeof (int), 1, file);
    fwrite (&(net -> learn_rate), sizeof (float), 1, file);
    fwrite (&(net -> dropout), sizeof (float), 1, file);
    fwrite (net -> neurons, sizeof (int), length, file);
    fwrite (net -> act_ids, sizeof (int), length - 1, file);

    for (int i = 0; i < length - 1; i++) {
        int size = layers[i] -> weights -> row * layers[i] -> weights -> col;
        fwrite (layers[i] -> weights -> ten, sizeof (float), size, file);
        size = layers[i] -> biases -> row * layers[i] -> biases -> col;
        fwrite (layers[i] -> biases -> ten, sizeof (float), size, file);
    }

    fclose(file);
}

mcl_network* mcl_network_import (const char *path)
{
    FILE *file = fopen (path, "rb");

    if (!file) {
        printf ("Could not import network.\n");
        return NULL;
    }

    int length;
    int cost_id;
    int learn_rate;
    int dropout;

    fread (&length, sizeof (int), 1, file);
    fread (&cost_id, sizeof (int), 1, file);
    fread (&learn_rate, sizeof (float), 1, file);
    fread (&dropout, sizeof (float), 1, file);

    int *neurons = malloc (sizeof (int) * length);
    int *act_ids = malloc (sizeof (int) * (length - 1));

    fread (neurons, sizeof (int), length, file);
    fread (act_ids, sizeof (int), length - 1, file);

    mcl_network *net = mcl_network_create (length, neurons);

    mcl_network_set_cost (net, cost_id);
    mcl_network_set_learn_rate (net, learn_rate);
    mcl_network_set_dropout (net, dropout);
    mcl_network_set_activations (net, act_ids);

    mcl_layer **layers = net -> layers;

    for (int i = 0; i < length - 1; i++) {
        int size = layers[i] -> weights -> row * layers[i] -> weights -> col;
        fread (layers[i] -> weights -> ten, sizeof (float), size, file);
        size = layers[i] -> biases -> row * layers[i] -> biases -> col;
        fread (layers[i] -> biases -> ten, sizeof (float), size, file);
    }

    fclose (file);

    return net;
}