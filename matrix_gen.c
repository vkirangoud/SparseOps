
#include "spmv.h"
#include <stdlib.h>

CSRMatrix generate_random_symmetric_csr(int n, double density) {
    int max_nnz = (int)(n * n * density);
    int* row_counts = calloc(n, sizeof(int));
    int* row_buffer = malloc(2 * max_nnz * sizeof(int));
    int* col_buffer = malloc(2 * max_nnz * sizeof(int));
    double* val_buffer = malloc(2 * max_nnz * sizeof(double));

    srand(42);
    int nnz = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            if (((double)rand() / RAND_MAX) < density) {
                double v = ((double)rand() / RAND_MAX) * 2 - 1;
                row_buffer[nnz] = i;
                col_buffer[nnz] = j;
                val_buffer[nnz++] = v;
                if (i != j) {
                    row_buffer[nnz] = j;
                    col_buffer[nnz] = i;
                    val_buffer[nnz++] = v;
                }
                row_counts[i]++;
                if (i != j) row_counts[j]++;
            }
        }
    }

    CSRMatrix A;
    A.n = n;
    A.nnz = nnz;
    A.values = malloc(nnz * sizeof(double));
    A.col_index = malloc(nnz * sizeof(int));
    A.row_ptr = malloc((n + 1) * sizeof(int));

    int* row_pos = calloc(n, sizeof(int));
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i)
        A.row_ptr[i + 1] = A.row_ptr[i] + row_counts[i];

    for (int k = 0; k < nnz; ++k) {
        int row = row_buffer[k];
        int dest = A.row_ptr[row] + row_pos[row]++;
        A.col_index[dest] = col_buffer[k];
        A.values[dest] = val_buffer[k];
    }

    free(row_counts);
    free(row_buffer);
    free(col_buffer);
    free(val_buffer);
    free(row_pos);
    return A;
}

void free_csr(CSRMatrix* A) {
    free(A->values);
    free(A->col_index);
    free(A->row_ptr);
}
