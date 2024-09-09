# No Clean Example Required: Resilient and Robust Defense Against Adversarial Attacks


## 
This is an implementation for No Clean Example Required: Resilient and Robust Defense Against Adversarial Attacks


<td><img src="figure/mainfig.png"></td>


## Adversarial Attack

Run the attack notebook:  


Open the Jupyter Notebook **`FGSM_attack.ipynb`** and run the cells step-by-step to perform the attack:
```
FGSM_attack.ipynb
```

## PGD-based Rectification

Run the rectification notebook:  

Open the Jupyter Notebook **`PGD-based_rectification.ipynb`** and run the cells step-by-step to perform the rectification:

```
PGD-based_rectification.ipynb
```

# Noise2Noise-based Defense

## Dependencies

* [PyTorch](https://pytorch.org/) (0.4.1)
* [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [NumPy](http://www.numpy.org/) (1.14.2)
* [Matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)
* [OpenEXR](http://www.openexr.com/) (1.3.0)

To install the latest version of all packages, run
```
pip3 install --user -r requirements.txt
```

## Dataset

Add your images to the `data/test` folder. Only a handful will do to visually inspect the denoiser performance.

## Training

See `python3 train.py --h` for list of optional arguments, or `examples/train.sh` for an example.

### Adversarial noise

```
python3 train.py \
  --train-dir ../data/train --train-size 5 \
  --valid-dir ../data/valid --valid-size 4 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 30 \
  --batch-size 1 \
  --loss l2 \
  --noise-type adv \
  --crop-size 0 \
  --cuda
```

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, provide `test.py` with a PyTorch model (`.pt` file) via the argument `--load-ckpt` and a test image directory via `--data`. The `--show-output` option specifies the number of noisy/denoised/clean montages to display on screen. To disable this, simply remove `--show-output`.

```
python3 test.py \
  --data ../data/test \
  --load-ckpt ../ckpts/adv/n2n.pt \
  --noise-type adv \
  --crop-size 0 \
  --show-output 1 \
  --cuda
```

See `python3 test.py --h` for list of optional arguments, or `examples/test.sh` for an example.

## Results

MNIST model was trained on an NVIDIA GeForce RTX 3090.

<center><img src="figure/MNIST_result.png" width="500"></center>
<center><img src="figure/LivDet_result.png" width="500"></center>


## References
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. [*Noise2Noise: Learning Image Restoration without Clean Data*](https://research.nvidia.com/publication/2018-07_Noise2Noise%3A-Learning-Image). Proceedings of the 35th International Conference on Machine Learning, 2018.  

[//]: # (* Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. [*Microsoft COCO: Common Objects in Context*]&#40;https://arxiv.org/abs/1405.0312&#41;. arXiv:1405.0312, 2014.)
* [Yusuke Uchida](https://yu4u.github.io/), [Keras implementation of Noise2Noise](https://github.com/yu4u/noise2noise).   


* [Joey Litalien](https://joeylitalien.github.io/), [PyTorch Implementation of Noise2Noise](https://github.com/joeylitalien/noise2noise-pytorch). 
