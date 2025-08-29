CC=gcc
CFLAGS=-march=native
LIBS=-lm

all: main

clean:
	rm -f *.o main

main.o: main.c machl.h
	${CC} -c main.c -o main.o ${CFLAGS}

machl.o: machl.h
	${CC} -c machl.c -o machl.o ${CFLAGS}

main: main.o machl.o
	${CC} main.o machl.o ${LIBS} -o main
