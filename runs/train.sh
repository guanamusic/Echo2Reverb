export CUDA_VISIBLE_DEVICES=0

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

CONFIG_PATH=configs/echo2reverb.json
VERBOSE="yes"

python train.py -c $CONFIG_PATH -v $VERBOSE
