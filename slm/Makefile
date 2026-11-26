# slm/Makefile -- llama2.c (TinyStories-15M) on the SA v2.4 / SISA v2.5 xclbins. Build on an RCE COMPUTE node:
#   sinteractive -c 8 -A cs2401 ; source ../env.sh ; make -C slm            (needs run.c next to this Makefile)
#   make run_cpu                 CPU reference build (plain run.c, -O3)
#   make slm_host                llama2.c + FPGA hook  (needs XRT)
#   make slm_sweep               decode-shape GEMM sweep (needs XRT)
#   make mocktest                no-board check against the csim kernels (no XRT needed; see test/)
SHELL := /bin/bash
XRT_INC ?= $(XILINX_XRT)/include
XRT_LIB ?= $(XILINX_XRT)/lib
CXXFLAGS ?= -std=c++17 -O2
XRT_LDFLAGS = -L$(XRT_LIB) -lxrt_coreutil -lpthread -Wl,--allow-shlib-undefined -lm

all: run_cpu slm_host slm_sweep

run_cpu: run.c
	gcc -O3 -o $@ run.c -lm

run_fpga.c: run.c patch_run.py
	python3 patch_run.py run.c $@

run_fpga.o: run_fpga.c
	gcc -O3 -DFPGA_HOOK -c run_fpga.c -o $@

fpga_matmul.o: fpga_matmul.cpp fpga_backend.h ../src/common.h
	g++ $(CXXFLAGS) -I$(XRT_INC) -c fpga_matmul.cpp -o $@

slm_host: run_fpga.o fpga_matmul.o
	g++ -o $@ run_fpga.o fpga_matmul.o $(XRT_LDFLAGS)

slm_sweep: sweep_main.cpp fpga_backend.h ../src/common.h
	g++ $(CXXFLAGS) -I$(XRT_INC) sweep_main.cpp -o $@ $(XRT_LDFLAGS)

mocktest:
	bash test/run_mocktest.sh

clean:
	rm -f run_cpu run_fpga.c run_fpga.o fpga_matmul.o slm_host slm_sweep
.PHONY: all clean mocktest
