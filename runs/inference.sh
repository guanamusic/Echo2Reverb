export CUDA_VISIBLE_DEVICES=0

DIR=echo2reverb
CONFIG_PATH=logs/$DIR/config.json
CHECKPOINT_PATH=logs/$DIR/checkpoint_399.pt
RIR_FILELIST_PATH=filelists/test.txt
EXPORT_MISC="no"
VERBOSE="yes"

python inference.py -c $CONFIG_PATH -ch $CHECKPOINT_PATH -rs $RIR_FILELIST_PATH -exp $EXPORT_MISC -v $VERBOSE
