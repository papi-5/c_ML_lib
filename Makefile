CC=gcc
CFLAGS=-march=native
LIBS=-lm
INC := -I include/

SRCDIR := src/
OBJDIR := build/
HEADIR := include/
BINDIR := bin/

SRC := main.c machl_export.c machl_net.c machl_layer.c machl_function.c machl_tensor.c
SRC := $(patsubst %.c,$(SRCDIR)%.c,$(SRC))
OBJ := $(patsubst $(SRCDIR)%.c,$(OBJDIR)%.o,$(SRC))
HEAD := $(shell find $(HEADIR) -name '*.h')

MAIN := $(BINDIR)main

all: $(MAIN)

compile: $(OBJ)

run: $(MAIN)
	./$(MAIN)

clean:
	rm -f $(OBJDIR)* $(BINDIR)*

$(MAIN): $(OBJ)
	$(CC) $(OBJ) -o $(MAIN) $(LIBS)

$(OBJDIR)%.o: $(SRCDIR)%.c $(HEAD)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
