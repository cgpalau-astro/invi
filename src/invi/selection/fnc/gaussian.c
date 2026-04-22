//----------------------------------------------------------------------------
//Multivariate Gaussian distribution

double *cholesky_decomposition(double *A) {
    double *L = malloc(D*D * sizeof(double));
    double s;

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < (i+1); j++) {
            s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * D + k] * L[j * D + k];
            L[i * D + j] = (i == j) ?
                            sqrt(A[i * D + i] - s) :
                            (1.0 / L[j * D + j] * (A[i * D + j] - s));
        }
    }
    return L;
}


double sqrt_det(double *A) {
    double product = 1.0;

    for (int i = 0; i < D; i++) {
        product *= A[i * D + i];
    }

    return product;
}


#define L(f,c) (A[D*f + c])
#define Li(f,c) (Ai[D*f + c])

double *InvL(double *A) {
    double sum = 0.0;
    double *Ai = calloc(D*D, sizeof(double));

    for (int j = 0; j < D; j++) {
        Li(j,j) = 1.0/L(j,j);
        for (int i = 0; i < j; i++) {
            for (int k = i; k <= j-1; k++) {
                sum -= L(j,k)*Li(k,i);
            }
        Li(j,i) = sum / L(j,j);
        sum = 0.0;
        }
    }

    return Ai;
}

#undef L
#undef Li
    

double *DotMtM(double *A) {
    double sum = 0.0;
    double *Am = calloc(D*D, sizeof(double));

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = j; k < D; k++) {
                sum += A[k*D + i] * A[k*D + j];
            }
            Am[i*D + j] = sum;
            sum = 0.0;
        }
    }

    return Am;
}
    

double exp_factor( double *x, double *mean, double *A ) {
    double xmu[D];

    for (int f = 0; f < D; f++) {
        xmu[f] = x[f] - mean[f];
    }

    double sum0 = 0.0, sum1 = 0.0;

    for (int f = 0; f < D; f++) {
        for (int c = 0; c < D; c++) {
            sum0 += A[D*f + c]*xmu[c];
        }
        sum1 += sum0*xmu[f];
        sum0 = 0.0;
    }

    return sum1;
}


double multivariate_normal(double *x, double *w, double *S) {

    double *CH = cholesky_decomposition(S);
    double sqrtdet = sqrt_det(CH);
    double *CHI = InvL(CH);
    double *Mi = DotMtM(CHI);
    double factor = exp_factor(x, w, Mi);

    free(CH);
    free(CHI);
    free(Mi);

    return 1.0/pow(2.0*PI, D/2) / sqrtdet * exp(-0.5*factor);

}
    
//----------------------------------------------------------------------------

double *add(double *A, double *B) {

    double *C = malloc(D*D * sizeof(double));

    for (int f = 0; f < D; f++) {
        for (int c = 0; c < D; c++) {
            C[D*f + c] =  A[D*f + c] + B[D*f + c];
        }
    }

    return C;
}

//----------------------------------------------------------------------------
//Gaussian convolution

double gaussian_convolution(struct WS ws, struct WS *volume_ob, int n_ob) {
    double sum = 0.0;
    double *C = malloc(D*D * sizeof(double));

    for (int j = 0; j < n_ob; j++) {
        C = add(ws.S, volume_ob[j].S);
        sum += multivariate_normal(ws.w, volume_ob[j].w, C);
        free(C);
    }

    return sum/(double)n_ob;
}
