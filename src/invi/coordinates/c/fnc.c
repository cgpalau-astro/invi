//General Includes
#include <stdio.h>
#include <stdlib.h>

//Math
#include <math.h> //-lm

#define PI 3.14159265358979323846
#define INF (1.0/0.0)

//----------------------------------------------------------------------------

double sgn(double x) { 
        if (x > 0.0) return 1.0;
        if (x < 0.0) return -1.0;
        return 0.0;
    }

//----------------------------------------------------------------------------
