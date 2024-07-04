# Style Transfer 3x3 Analysis

<!-- Badges -->
<p>
  <a href="https://github.com/ayakiri/cov-project/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/ayakiri/cov-project" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/ayakiri/cov-project" alt="last update" />
  </a>
</p>
<!-- Table of Contents -->


## Introduction
Style transfer is a technique in the field of Computer Vision that allows for transferring the appearance of one image onto another while preserving its original shape and layout. Thanks to advancements in neural networks and their ability to extract complex visual features, we can make, for example, an ordinary photo look like it was painted by a famous artist. Style transfer is popular in digital art, graphics, and photo editing.

The choice of appropriate tools is very important to achieve the best results in style transfer. We have two main tools: encoders and optimizers. Our goal is to find the best combination of encoder and optimizer to create the highest quality images.


## Dataset
The dataset "Images for Style Transfer" provided by Soumik Rakshit contains 27 example style images and 10 content images.
Source: https://www.kaggle.com/datasets/soumikrakshit/images-for-style-transfer

## Setup
Clone repository:
```bash
git clone https://github.com/ayakiri/cov-project.git
```
Install requirements:
```bash
poetry install
```

## Sample run
Run
```bash
python.exe .\main.py -fp "path/to/content/img.jpg" -sp "path/to/style/img.jpg" -op "path/to/save/img.jpg"
```

There are many arguments and parameters that we can change:
* **file-path** (-fp) - path to content image
* **output-path** (-op) - path to transformed image to be saved
* **style-path** (-sp) - path to style image
* **steps** (-s) - number of steps
* **encoder** (-e) - encoder ("vgg19", "resnet50", "inception_v3")
* **optimizer** (-o) - optimizer ("lbfgs", "adam", "sgd")

## Pre-commits
Install pre-commits
https://pre-commit.com/#installation

To use
```
pre-commit run --all-files
```

## Adding python packages
Dependencies are handeled by `poetry` framework, to add new dependency run
```
poetry add <package_name>
```

<!-- Contributing -->
## :wave: Contributors

<a href="https://github.com/ayakiri/cov-project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ayakiri/cov-project" />
</a>
