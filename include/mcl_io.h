#ifndef MCL_IO_H
#define MCL_IO_H

#include "mcl_network.h"

void mcl_network_print_meta (mcl_network *net);

void mcl_network_export (const char *path, mcl_network *net);

mcl_network* mcl_network_import (const char *path);

#endif