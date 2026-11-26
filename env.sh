#!/usr/bin/env bash
# env.sh -- Vitis 2025.1 + XRT + U50 platform on the RCE cluster (same paths as the course starter kit).
#   source ./env.sh
source /home/Xilinx/2025.1/Vitis/settings64.sh
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms:/home/Xilinx/opt/xilinx/platforms
export PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1
export PART=xcu50-fsvh2104-2-e
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-2100@rce.iiit.ac.in}"
