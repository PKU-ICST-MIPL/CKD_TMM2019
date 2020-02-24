export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib:/usr/local/cuda-8.0/lib64
alias python="/usr/bin/python2.7"

cd ./stage3
python test.py 