set -ex
ROOT_DIR=`dirname $0`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_DIR
export FMHALIB_PATH=$ROOT_DIR/libfmha.so

nvcc ${ROOT_DIR}/test_fmha_so.cc -o ${ROOT_DIR}/test_fmha_so -L$PWD -lfmha -ldl
${ROOT_DIR}/test_fmha_so
