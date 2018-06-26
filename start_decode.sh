export PYTHONPATH=`pwd`
MODEL=$1
python model/decode.py $MODEL >& ../log/decode_log &

