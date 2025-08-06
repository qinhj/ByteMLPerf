# User Guide

## Env

* prepare docker container

```bash
# download and load docker image
wget https://bj.bcebos.com/v1/klx-public/cyliu/release/bytedance/release/0117/xmlir_ubuntu_2004_x86_64.v0.29b.tar.gz
docker load -i xmlir_ubuntu_2004_x86_64.v0.29b.tar.gz

# create new docker container
docker run -itd \
    --name=bytemlperf_test \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    $(for dev in /dev/xpu*; do echo --device=$dev:$dev; done) \
    -v $HOME:/workspace -w /workspace \
    --restart=always \
    iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.29b
```

* prepare klx dependencies (within docker container)

```bash
docker exec -it bytemlperf_test /bin/bash

# download klx dependencies
wget -O xpytorch-cp38-torch201-ubuntu2004-x64.run.0708 https://klx-public.bj.bcebos.com/v1/cyliu/release/bytedance/xmlir_installer/xpytorch-cp38-torch201-ubuntu2004-x64.run.0708_round_mode_3?authorization=bce-auth-v1%2FALTAKm6sVAVJZnS4GoT8hKS6Ls%2F2025-07-08T08%3A37%3A21Z%2F-1%2Fhost%2Fff3554438d44dac8aeb8ed17330452f60bed0a57e7fe399fb0df861011b70cfa
wget -O flash_mla-1.0.0+9636021-py3-none-any.whl https://klx-public.bj.bcebos.com/v1/cyliu/release/bytedance/flash_mla/0611/flash_mla-1.0.0%2B9636021-py3-none-any.whl?authorization=bce-auth-v1%2FALTAKm6sVAVJZnS4GoT8hKS6Ls%2F2025-06-12T03%3A22%3A24Z%2F-1%2Fhost%2F5c3db78f671d711c10a1d8714bbf234996e0bc74a188da386a01ca08f0c4fec4
wget -O xtorch_ops-0.0.0-cp38-cp38-linux_x86_64.whl https://klx-public.bj.bcebos.com/v1/cyliu/release/bytedance/flash_mla/0611/xtorch_ops-0.0.0-cp38-cp38-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKm6sVAVJZnS4GoT8hKS6Ls%2F2025-06-12T03%3A24%3A18Z%2F-1%2Fhost%2Fc2a809921b0b2f2d5233539cca3163ddd74de5a6768522cebe7d2f926be61682
wget -O bmm_merged_output0318-moe.plan https://klx-public.bj.bcebos.com/v1/cyliu/release/bytedance/tune_files/bmm_merged_output0318-moe.plan?authorization=bce-auth-v1%2Fc46978d3cefa492f827002c8c050fc67%2F2025-03-19T08%3A52%3A28Z%2F-1%2Fhost%2F3abf27fea70f26c4d2edaa0dc07470b5c1adf728ca6ad36d9ccd894d29616342
wget -O gemm_merged_output0703_2.plan https://klx-sdk-release-public.su.bcebos.com/v1/chenyuezi/gemm_merged_output0703_2.plan?authorization=bce-auth-v1%2FALTAKm6sVAVJZnS4GoT8hKS6Ls%2F2025-07-07T07%3A58%3A34Z%2F-1%2Fhost%2F87e9c3276facabfdecd7326212e107055eb108b036088715f9dd004d8d256cc8

# install klx dependencies
bash xpytorch-cp38-torch201-ubuntu2004-x64.run.0708
pip3 install flash_mla-1.0.0+9636021-py3-none-any.whl
pip3 install xtorch_ops-0.0.0-cp38-cp38-linux_x86_64.whl
```

## Test

* within docker container

```bash
# clone source code
git clone -b klx_dev https://github.com/suisiyuan/ByteMLPerf_KLX.git
cd ByteMLPerf_KLX/byte_micro_perf
# install requirements
pip3 install -r requirements.txt
# quick test
env MASTER_PORT=8899 MASTER_ADDR=127.0.0.1 XDNN_USE_FAST_SWISH=true \
    FAST_SWIGLU_ENABLE=1 CUDART_DUMMY_REGISTER=1 XPU_FORCE_USERMODE_LAUNCH=1 \
    XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 XMLIR_BMM_TO_MOE=1 \
    XBLAS_FC_AUTOTUNE_FILE=/workspace/gemm_merged_output0703_2.plan \
    XBLAS_MOE_FC_AUTOTUNE_FILE=/workspace/bmm_merged_output0318-moe.plan \
    XBLAS_FC_PRINT_CACHE_HITS=1 XBLAS_MOE_FC_PRINT_CACHE_HITS=1 \
    python launch.py --hardware KLX --task_dir workloads/basic | tee byte_micro_perf.klx.log # --task_dir workloads/llm
```
