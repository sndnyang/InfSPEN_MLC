# Multi-Label Classification with Inference Structured Prediction Energy Net

Under construction


## Changelog

- 20210527   
     - Loaded the Tensorflow model's weights (copied.tar) into the pytorch model, https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28 
     - The models of TF and PyTorch can give the same outputs.  see infnet-tf2pytorch.ipynb
     - Training still is different.
- 20210521 first trial
     - Approximate Inference Network, failed.
     - Improved joint training, it works under some unexpected cases.

## 

This implementation is based on following repos:

1. https://github.com/davidBelanger/SPEN  Official, Theano Code
2. https://github.com/TheShadow29/infnet-spen, TF code (find datasets from here)
3. https://github.com/tyliupku/Arbitrary-Order-Infnet, Improved version

## installation

python 3.6

pip install -r requirements.txt

## usage

all hyper-parameters are written in code

Original one:

python inf-spe.py

Joint Inf SPEN:

python jispen.py


## Tips

https://github.com/TheShadow29/infnet-spen, TF code (find datasets from here)

Here are some mistakes that you maybe make (Thanks for Kalpesh's feedback)   

- Not having a classification threshold and assuming it to be 0.5.

- Not pre-training b_i jointly with F(x) in the first stage, and misunderstanding the scheme used to load the initial A(x) parameters.

- Calculating sum(hinge loss objective) rather than mean(hinge loss objective).

- Calculating label averaged F-1 scores rather than example averaged F-1 scores
