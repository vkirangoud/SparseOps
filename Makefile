CC = gcc
SRC = spmv.c matrix_gen.c utils.c dense_ref.c
HDR = spmv.h
OUT = spmv
CFLAGS = -O3 -march=native -fopenmp

# Use MODE=avx2 or MODE=avx512
MODE ?= avx2

ifeq ($(MODE), avx2)
    CFLAGS += -mavx2 -mfma
    CDEFS  = -DUSE_AVX2
else ifeq ($(MODE), avx512)
    CFLAGS += -mavx512f -mavx512dq -mfma
    CDEFS  = -DUSE_AVX512
else
    $(error Unknown MODE specified: use MODE=avx2 or MODE=avx512)
endif

all: $(OUT)

$(OUT): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(CDEFS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT) *.o *.csv
.PHONY: all clean
run: $(OUT)
	./$(OUT)
run_ref: $(OUT)
	./$(OUT) ref
run_avx2: $(OUT)
	./$(OUT) avx2
run_avx512: $(OUT)
	./$(OUT) avx512
run_all: $(OUT)
	./$(OUT) ref
	./$(OUT) avx2
	./$(OUT) avx512
run_bench: $(OUT)
	./$(OUT) bench	