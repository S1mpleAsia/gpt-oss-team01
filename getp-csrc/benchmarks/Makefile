HIPCC ?= hipcc
HIPFLAGS ?= -O3 --std=c++17 --offload-arch=gfx90a
INCLUDES ?= -I..

GEMM_SRCS := gemm_bench.cpp
FLASH_ATTN_SRCS := \
	flash_attn_bench.cpp \
	flash_attn_workspace.cpp \
	../src/flash_attn_hip.cpp \
	../src/tensor.cpp

GEMM_LIBS ?= -lrocblas
FLASH_ATTN_FLAGS ?= -include config_types_fwd.hpp

TARGETS := gemm_bench flash_attn_bench

.PHONY: all clean flash_attn_bench gemm_bench

all: gemm_bench

gemm_bench: $(GEMM_SRCS)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES) $(GEMM_SRCS) -o $@ $(GEMM_LIBS)

flash_attn_bench: $(FLASH_ATTN_SRCS)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES) $(FLASH_ATTN_FLAGS) $(FLASH_ATTN_SRCS) -o $@

clean:
	rm -f $(TARGETS)
