# SISA-on-FPGA project Makefile. Run builds on a COMPUTE node (sinteractive / sbatch), never the login node.
#   make csim         g++ functional test of both kernels (no Vitis needed, seconds)
#   make hls-sa       Vitis HLS csynth -> build/gemm_sa.xo        (also: hls-sisa)
#   make xclbin-sa    v++ link -> build/sa.xclbin  (~1 h)         (also: xclbin-sisa)
#   make host         build the XRT host
#   make model        first-principles speedup model (scripts/fusion_model.py)
SHELL := /bin/bash
BUILD := build
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1

.PHONY: cosim-sisa csim hls-sa hls-sisa xclbin-sa xclbin-sisa host model clean
csim:
	@mkdir -p $(BUILD)
	g++ -std=c++17 -O2 -ffp-contract=off -DSIM_COUNT -Wno-unknown-pragmas \
	    -Itest/shim -Isrc test/tb.cpp src/gemm_sa.cpp src/gemm_sisa.cpp -o $(BUILD)/tb
	$(BUILD)/tb | tail -3

hls-sa:
	@mkdir -p $(BUILD)
	sed 's#@ROOT@#$(CURDIR)#g' configs/hls_sa.cfg > $(BUILD)/hls_sa.cfg
	source ./env.sh && v++ -c --mode hls --config $(CURDIR)/$(BUILD)/hls_sa.cfg --work_dir $(CURDIR)/$(BUILD)/hls_sa
	cp $(BUILD)/hls_sa/*.xo $(BUILD)/gemm_sa.xo 2>/dev/null || true
hls-sisa:
	@mkdir -p $(BUILD)
	sed 's#@ROOT@#$(CURDIR)#g' configs/hls_sisa.cfg > $(BUILD)/hls_sisa.cfg
	source ./env.sh && v++ -c --mode hls --config $(CURDIR)/$(BUILD)/hls_sisa.cfg --work_dir $(CURDIR)/$(BUILD)/hls_sisa
	cp $(BUILD)/hls_sisa/*.xo $(BUILD)/gemm_sisa.xo 2>/dev/null || true

# RTL co-simulation of gemm_sisa on a tiny case (reproduces board failures without using a board slot).
# Needs a finished `make hls-sisa` (csynth) first.
cosim-sisa:
	@mkdir -p $(BUILD)
	sed 's#@ROOT@#$(CURDIR)#g' configs/hls_sisa_cosim.cfg > $(BUILD)/hls_sisa_cosim.cfg
	source ./env.sh && vitis-run --mode hls --cosim --config $(CURDIR)/$(BUILD)/hls_sisa_cosim.cfg --work_dir $(CURDIR)/$(BUILD)/hls_sisa

xclbin-sa:
	source ./env.sh && v++ -l -t hw --platform $(PLATFORM) --config configs/connectivity_sa.cfg \
	    $(BUILD)/gemm_sa.xo -o $(BUILD)/sa.xclbin
xclbin-sisa:
	source ./env.sh && v++ -l -t hw --platform $(PLATFORM) --config configs/connectivity_sisa.cfg \
	    $(BUILD)/gemm_sisa.xo -o $(BUILD)/sisa.xclbin

host:
	@mkdir -p $(BUILD)
	source /opt/xilinx/xrt/setup.sh && g++ -std=c++17 -O2 host/host.cpp -o $(BUILD)/host \
	    -I$$XILINX_XRT/include -L$$XILINX_XRT/lib -lxrt_coreutil -lpthread -Wl,--allow-shlib-undefined

model:
	python3 scripts/fusion_model.py

clean:
	rm -rf $(BUILD)
