# Installation instructions
**a. Create a conda virtual environment and activate it.**
```shell
conda create -n rfcrnl python=3.7 -y
conda activate rfcrnl
```

**1. Install Tensorflow following the [official instructions](https://www.tensorflow.org/).**

Our code has been tested under Tensorflow=1.13.1 and CUDA=10.2.
```shell
conda install tensorflow-gpu==1.13.1
```


**2. Install other requirements.**
```shell
pip install -r requirements.txt
```

**3. Install third_party.**
```shell
bash compile_op.sh
```