CC = gcc
CFLAGS = -march=native -Wall -Wextra
LIBS = -lm
INC = -I include/

SRCDIR = src/
OBJDIR = build/
BINDIR = bin/

SRC = $(wildcard $(SRCDIR)*.c)
OBJ = $(patsubst $(SRCDIR)%.c, $(OBJDIR)%.o, $(SRC))
MAIN = $(BINDIR)main

.PHONY: all compile run clean

all: $(MAIN)

compile: $(OBJ)

run: $(MAIN)
	./$(MAIN)

clean:
	rm -f $(OBJDIR)* $(BINDIR)*

$(MAIN): $(OBJ) | $(BINDIR)
	$(CC) $(OBJ) -o $(MAIN) $(LIBS)

$(OBJDIR)%.o: $(SRCDIR)%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)