## Introduction
This is the source code of our IEEE TMM 2019 paper "CKD: Cross-task Knowledge Distillation for Text-to-image Synthesis" and ACM MM 2018 paper "Text-to-image Synthesis via Symmetrical Distillation Networks". Please cite the following papers if you use our code.

Mingkuan Yuan and Yuxin Peng, "CKD: Cross-task Knowledge Distillation for Text-to-image Synthesis", IEEE Transactions on Multimedia (TMM), DOI:10.1109/TMM.2019.2951463, Nov. 2019. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201920)

Mingkuan Yuan and Yuxin Peng, "Text-to-image Synthesis via Symmetrical Distillation Networks", 26th ACM Multimedia Conference (ACM MM), pp. 1407-1415, Seoul, Korea, Oct. 22-26, 2018. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201820)

## Environment
CUDA 8.0

Python 2.7.12

TensorFlow 1.2.1

## Data Preparation
Download the preprocessed char-CNN-RNN text embeddings and filename lists for [birds](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE), which should be saved in data/cub/

Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data and extract them to data/cub/images/

Run the following command:

    - python data_preprocess.py

## Model Preparation
Download the pretrained [VGG19](https://drive.google.com/file/d/0B_B_FOgPxgFLRjdEdE9NNTlzUWc/view) model for Stages I and II which should be saved in models/

Download the pretrained [Show-and-Tell](https://drive.google.com/file/d/0B3laN3vvvSD2T1RPeDA5djJ6bFE/view) model and its [index](https://github.com/KranthiGV/Pretrained-Show-and-Tell-model/blob/master/model.ckpt-2000000.index) to data/models/img2txt_model/

Download the [Inception score](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) model and unzip it in models/inception_score/ for evaluating the trained model

## Usage
    - run 'sh train_all.sh' to train the models of three stages
    - run 'sh test_stage3.sh' to evaluate the final inception score
    
## Our Related Work
If you are interested in text-to-image synthesis, you can check our recently published papers about it:

Mingkuan Yuan and Yuxin Peng, "Bridge-GAN: Interpretable Representation Learning for Text-to-image Synthesis", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), DOI:10.1109/TCSVT.2019.2953753, Nov. 2019. [[pdf]](http://59.108.48.34/tiki/download_paper.php?fileId=201922)

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.