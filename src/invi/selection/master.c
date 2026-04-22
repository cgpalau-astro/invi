#include <stdlib.h>
#include <stdio.h>

//-lm
#include <math.h>

#define PI 3.14159265358979323846

//Number of dimensions
#define D 6

//----------------------------------------------------------------------------

int count_file_lines(char *file) {

    FILE *fr = fopen(file, "r");

    //Check whether the file exists
    if (fr == NULL) {
        printf("FileNotFoundError: %s\n", file);
        return -1;
    }

    //Count number lines
    int number_lines = 0;
    char c;
    for (c = getc(fr); c != EOF; c = getc(fr)) {
        if (c == '\n') { number_lines += 1; }
    }

    fclose(fr);

    return number_lines;
}

//----------------------------------------------------------------------------
//Definition structs

struct WS {
    double *w;
    double *S;
};

struct WS *Init_WS(int n) {

    struct WS *ws = malloc(n * sizeof(struct WS));

    for (int i = 0; i < n; i++) {
        ws[i].w = calloc(D, sizeof(double));
        ws[i].S = calloc(D*D, sizeof(double));
    }

    return ws;
}

void free_WS(struct WS *ws, int n) {

    for (int i = 0; i < n; i++) {
        free(ws[i].w);
        free(ws[i].S);
    }

    free(ws);
}

//----------------------------------------------------------------------------

//Gaussian convolution
#include "fnc/gaussian.c"

//Load data
#include "fnc/data.c"

//Core functions
#include "fnc/core.c"

//----------------------------------------------------------------------------

#include "fnc/test.c"

int main() {
    test_gauss();
}

//Compile shared object
//gcc -Wall -O3 -lm -shared -o master.so -fPIC master.c

//Run test:
//gcc -Wall -O3 master.c -o /tmp/test -lm; /tmp/test
