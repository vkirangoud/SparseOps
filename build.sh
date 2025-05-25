
#!/bin/bash
set -e
make clean
make MODE=avx2
./spmv
