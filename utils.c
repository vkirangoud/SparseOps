
#include "spmv.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double* rand_vector(int n) {
    double* x = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        x[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    return x;
}

void run_and_log(const char* label, void (*spmv)(const CSRMatrix*, const double*, double*), const CSRMatrix A, const double* x, double* y) {
    double* y_ref = calloc(A.n, sizeof(double));
    spmv_dense(&A, x, y_ref);

    double start = wall_time();
    spmv(&A, x, y);
    double end = wall_time();

    double err = 0.0;
    for (int i = 0; i < A.n; ++i)
        err += fabs(y[i] - y_ref[i]);
    if (err > 1e-8)
        printf("[Warning] %s result mismatch: total error = %g ", label, err);

    FILE* f = fopen("spmv_benchmark.csv", "a");
    fprintf(f, "%s,%d,%d,%f ", label, A.n, A.nnz, end - start);
    fclose(f);

    free(y_ref);
}
