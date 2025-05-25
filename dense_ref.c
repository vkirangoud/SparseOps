
#include "spmv.h"
#include <string.h>

void spmv_dense(const CSRMatrix* A, const double* x, double* y) {
    memset(y, 0, A->n * sizeof(double));
    for (int i = 0; i < A->n; ++i) {
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
            int j = A->col_index[k];
            y[i] += A->values[k] * x[j];
        }
    }
}
