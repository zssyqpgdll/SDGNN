export PYTHONPATH=`pwd`
MODEL=$1
python -u training_ptr_gen/decode.py $MODEL

