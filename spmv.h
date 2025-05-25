// File: spmv.h
#ifndef SPMV_H
#define SPMV_H

#include <stdio.h>

typedef struct {
    int n;
    int nnz;
    double* values;
    int* col_index;
    int* row_ptr;
} CSRMatrix;

CSRMatrix generate_random_symmetric_csr(int n, double density);
void sort_csr_rows(CSRMatrix* A);
void free_csr(CSRMatrix* A);

void spmv_naive(const CSRMatrix* A, const double* x, double* y);
void spmv_omp_buffered(const CSRMatrix* A, const double* x, double* y);
void spmv_avx2_buffered(const CSRMatrix* A, const double* x, double* y);
void spmv_avx512_buffered(const CSRMatrix* A, const double* x, double* y);
void spmv_dense(const CSRMatrix* A, const double* x, double* y);

double* rand_vector(int n);
void run_and_log(const char* label, void (*spmv)(const CSRMatrix*, const double*, double*), const CSRMatrix A, const double* x, double* y);

#endif // SPMV_H
