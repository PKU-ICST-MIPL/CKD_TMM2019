export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib:/usr/local/cuda-8.0/lib64
alias python="/usr/bin/python2.7"

cd ./stage1
python main.py

cd ../stage2
cp ../stage1/result/cub/models/model_0100.ckpt.* ./pretrained_model/
python main.py

cd ../stage3
cp ../stage2/result/cub/models/model_0080.ckpt.* ./pretrained_model/
python main.py