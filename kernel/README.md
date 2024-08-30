# environment
```
conda create -n kittens python=3.11
conda activate kittens
conda install cuda==12.4.0 -c nvidia
conda install cudnn=9.1.1.17
conda install cmake=3.26.4
pip install nvidia_cudnn_frontend==1.6.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install ninja=1.10.2

rm -rf $(python -c "import sys; import os; print(os.path.dirname(sys.executable)+'/ld')")

```

# conv2d
[The Indirect Convolution Algorithm](https://arxiv.org/pdf/1907.02129)

[BP-Im2col: Implicit Im2col Supporting AI Backpropagation on Systolic Arrays*](https://arxiv.org/pdf/2209.09434)

[Characterizing and Demystifying the Implicit Convolution Algorithm on Commercial Matrix-Multiplication Accelerators](https://arxiv.org/pdf/2110.03901)

[MEC: Memory-efficient Convolution for Deep Neural Network](https://arxiv.org/pdf/1706.06873)

[Performance Evaluation of cuDNN Convolution Algorithms on NVIDIA Volta GPUs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8721631)

# cuda
[WhoSays Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production](https://arxiv.org/abs/2211.10017)