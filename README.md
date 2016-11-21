# Envrionment

- OS: Ubuntu14.04
- Python: 3.4.5
- TensorFlow: 0.10.0
- TFLearn: 0.2.1

#  preparation

It needs `pip`(or `pip3`). So you prepare `pip`(for example, install pyenv) 

`pip install -r requirement.txt`

# Install Tensorflow

You need to install tensorflow individually.

```
# Ubuntu/Linux 64-bit, CPU only, Python 3.4
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
# Requires CUDA toolkit 7.5 and CuDNN v5. For other versions, see "Install from sources" below.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl
```

`pip install --upgrade $TF_BINARY_URL`


# Train

If you want to change the number of mixture in Gaussian distribution,
change `mixture_num`.

If you want to change the dimention of latent variable,
change `lantent_dim`

`python train.py`

# generator

You must set the same `mixture_num` and `lantent_dim` as training.
After this script, It generates `vae_std.png`.

`python generator.py`