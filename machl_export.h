#ifndef MACHL_EXPORT_H
#define MACHL_EXPORT_H

#include "machl_net.h"

int digits (int n);

int mostDigits (int *arr, int n);

char* intString (int n);

void printNetMeta (Net *net);

void exportNet (const char *path, Net *net);

Net* importNet (const char *path);

#endif