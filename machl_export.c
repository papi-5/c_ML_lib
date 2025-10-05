#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "machl_tensor.h"
#include "machl_function.h"
#include "machl_layer.h"
#include "machl_net.h"
#include "machl_export.h"

int digits (int n)
{
    int dig = 0;

    while (n) {
        n /= 10;
        dig++;
    }

    return dig;
}

int mostDigits (int *arr, int n)
{
    int dig = 0;

    for (int i = 0; i < n; i++) {
        int tmp = 0;
        int num = arr[i];

        tmp = digits (arr[i]);

        if (tmp > dig)
            dig = tmp;
    }

    return dig;
}

char* intString (int n)
{
    int dig = digits (n);
    int tmp = (int)pow (10, dig - 1);
    char *str = malloc (sizeof (char) * (dig + 4));
    str = "%";

    for (int i = 1; i < dig + 1; i++) {
        str[i] = (char)(n % tmp) - '0';
        n %= tmp;
        tmp /= 10;
    }

    str[dig + 1] = '\0';

    strcat (str, "d ");
    
    return str;
}

void printNetMeta (Net *net)
{
    int length = net -> numOfLayers;
    int dig = mostDigits(net -> neurons, length);
    char *str = intString (dig);

    printf ("Number of layers: %d\n\n", net -> numOfLayers);

    printf (" ");
    for (int i = 0; i < length; i++)
        printf (str, i);
    printf ("\n\n");

    printf (" ");
    for (int i = 0; i < length; i++)
        printf (str, net -> neurons[i]);
    printf ("\n\n");

    printf (" ");
    for (int i = 0; i < dig + 1; i++)
        printf (" ");
    for (int i = 0; i < length - 1; i++)
        printf (str, net -> layerActFunctions[i]);
    printf ("\n\n");

    printf ("Network cost function: %d\n", net -> netCostFunction);

    printf ("Learning rate: %f\n", net -> learnRate);

    printf ("Dropout: %f\n\n", net -> dropout);
}

void exportNet (const char *path, Net *net)
{

}