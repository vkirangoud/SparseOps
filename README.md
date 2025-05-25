# Symmetric Sparse Matrix-Vector Multiplication (SpMV) Benchmark

This repository contains optimized implementations of symmetric Sparse Matrix-Vector Multiplication (SpMV) using CSR format:
- Naive version
- OpenMP-parallel buffered
- AVX-512 vectorized
- Dense reference for validation

## Build and Run

```bash
chmod +x build.sh
./build.sh
```

## Requirements

- GCC with AVX-512 and OpenMP support

To **sort rows** of a CSR (Compressed Sparse Row) matrix by column indices within each row (which improves performance in many sparse algorithms), you'll need to sort the `col_index` and `values` arrays within each row range defined by `row_ptr`.

Here's how you can add row-wise sorting to your CSR matrix (safely, in-place):

---

### 🔧 **Function: Sort CSR Rows by Column Indices**

Add this to `matrix_gen.c` or a new utility file:

```c
#include <stdlib.h>

typedef struct {
    int col;
    double val;
} Entry;

void sort_csr_rows(CSRMatrix* A) {
    for (int i = 0; i < A->n; ++i) {
        int start = A->row_ptr[i];
        int end = A->row_ptr[i + 1];
        int len = end - start;
        Entry* row = malloc(len * sizeof(Entry));

        for (int j = 0; j < len; ++j) {
            row[j].col = A->col_index[start + j];
            row[j].val = A->values[start + j];
        }

        // Sort by column index
        qsort(row, len, sizeof(Entry), [](const void* a, const void* b) {
            return ((Entry*)a)->col - ((Entry*)b)->col;
        });

        for (int j = 0; j < len; ++j) {
            A->col_index[start + j] = row[j].col;
            A->values[start + j] = row[j].val;
        }

        free(row);
    }
}
```

---

### 📌 Usage

Call this function **after CSR generation** in your `main()`:

```c
CSRMatrix A = generate_random_symmetric_csr(n, 0.01);
sort_csr_rows(&A);  // 🔁 Sort before benchmarking
```

---

### ✅ Benefits

* Ensures **predictable memory access** for vectorization.
* Helps with **numerical reproducibility**.
* Required for some formats (like CSR-based SpMV with duplicate handling).

Would you like me to integrate this into your existing codebase and update the ZIP file?


sort_csr_rows(&A);  // 🔁 Sort before benchmarking 
          //✅ Benefits 
            // Ensures predictable memory access for vectorization.
            //Helps with numerical reproducibility. 
            // Required for some formats (like CSR-based SpMV with duplicate handling).