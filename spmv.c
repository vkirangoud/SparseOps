
#include "spmv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

void spmv_naive(const CSRMatrix* A, const double* x, double* y) {
    memset(y, 0, A->n * sizeof(double));
    for (int i = 0; i < A->n; ++i)
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            y[i] += A->values[k] * x[A->col_index[k]];
}



void spmv_omp_buffered(const CSRMatrix* A, const double* x, double* y) {
    int n = A->n;
    memset(y, 0, n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            sum += A->values[k] * x[A->col_index[k]];
        y[i] = sum;
    }
}
#ifdef USE_AVX512
void spmv_avx512_buffered(const CSRMatrix* A, const double* x, double* y) {
    int n = A->n;
    memset(y, 0, n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m512d vsum = _mm512_setzero_pd();
        int k;
        for (k = A->row_ptr[i]; k + 7 < A->row_ptr[i + 1]; k += 8) {
            __m512d vval = _mm512_loadu_pd(&A->values[k]);
            int idx[8];
            for (int j = 0; j < 8; ++j) idx[j] = A->col_index[k + j];
            __m512d vx = _mm512_i32gather_pd(_mm256_loadu_si256((__m256i*)idx), x, 8);
            vsum = _mm512_fmadd_pd(vval, vx, vsum);
        }
        double sum = _mm512_reduce_add_pd(vsum);
        for (; k < A->row_ptr[i + 1]; ++k)
            sum += A->values[k] * x[A->col_index[k]];
        y[i] = sum;
    }
}
#elif defined(USE_AVX2)
void spmv_avx2_buffered(const CSRMatrix* A, const double* x, double* y) {
    int n = A->n;
    memset(y, 0, n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m256d vsum = _mm256_setzero_pd();
        int k;
        for (k = A->row_ptr[i]; k + 3 < A->row_ptr[i + 1]; k += 4) {
            __m256d vval = _mm256_loadu_pd(&A->values[k]);
            int idx[4];
            for (int j = 0; j < 4; ++j) idx[j] = A->col_index[k + j];
            __m256d vx = _mm256_i32gather_pd(x, _mm_loadu_si128((__m128i*)idx), 8);
            vsum = _mm256_fmadd_pd(vval, vx, vsum);
        }
        double sum[4];
        _mm256_storeu_pd(sum, vsum);
        double total = sum[0] + sum[1] + sum[2] + sum[3];
        for (; k < A->row_ptr[i + 1]; ++k)
            total += A->values[k] * x[A->col_index[k]];
        y[i] = total;
    }
}
#else
spmv_naive_ref(const CSRMatrix* A, const double* x, double* y) {
    int n = A->n;
    memset(y, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
            y[i] += A->values[k] * x[A->col_index[k]];
        }
    }
}
#endif


int main() {
    FILE* f = fopen("spmv_benchmark.csv", "w");
    fprintf(f, "method,n,nnz,time_sec ");
    fclose(f);

    for (int n = 1000; n <= 10000; n += 1000) {
        CSRMatrix A = generate_random_symmetric_csr(n, 0.01);

        double* x = rand_vector(n);
        double* y = calloc(n, sizeof(double));

        run_and_log("naive", spmv_naive, A, x, y);
        run_and_log("openmp_buffered", spmv_omp_buffered, A, x, y);
       #if defined(USE_AVX2)
        run_and_log("avx2_buffered", spmv_avx2_buffered, A, x, y);
       #elif defined(USE_AVX512)
        run_and_log("avx512_buffered", spmv_avx512_buffered, A, x, y);
        #endif
        

        free(x);
        free(y);
        free_csr(&A);
    }

    return 0;
}
