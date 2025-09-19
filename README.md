# MCL-SV
This repository contains official pytorch implementation for “Mixture Consistency Learning for Robust Speaker Verification in Noisy Environments” paper.


# Abstract
![overall](https://github.com/user-attachments/assets/be9a710a-7458-40d2-868a-85ad6dfdc4ee)

Speaker verification (SV) systems often experience performance degradation in real-world environments due to diverse noise.
A common solution is to add a speech enhancement (SE) frontend; however, most SE modules are trained to reconstruct predefined ``clean'' reference signals that still contain remaining noise and channel variability.
This can lead the SE module to restore non-discriminative factors as well, thereby degrading SV performance.
To address this limitation, we propose a Mixture Consistency Learning–based Speaker Verification (MCL-SV) system.
Instead of relying on reference signals, MCL-SV enforces that the sum of the separated speaker and background representations matches the original input mixture, so that the enhanced signal is aligned with the input itself in a self-supervised manner. 
Experiments on multiple datasets show that MCL-SV consistently outperforms strong baselines and achieves state-of-the-art results among recent noise-robust SV systems.


Our experimental code was modified based on [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).


# Data
The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets were used for training and test.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```
The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt), and the test lists for VoxCeleb1 can be downloaded from [here](https://mm.kaist.ac.kr/datasets/voxceleb/index.html#testlist). 


For data augmentation, the following script can be used to download and prepare.
```
python3 ./dataprep.py --save_path data --augment
```

We also performed an out-of-domain evaluation using [Nonspeech 100](), [VoxSRC23](), and [VC-Mix]() datasets.

Each dataset must be downloaded in advance for training and testing, and its path must be mapped to the docker environment.

# Environment
Docker image (nvcr.io/nvidia/pytorch:23.07-py3) of Nvidia GPU Cloud was used for conducting our experiments.

Make docker image and activate docker container.
```
./docker/build.sh
./docker/run.sh
```

Note that you need to modify the mapping path before running the 'run.sh' file.

# Training

- on a single GPU
```
python3 ./main.py
```

- on multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1 python3 ./main.py --distributed
```
Use --distributed flag to enable distributed training.

# Test

```
python3 ./main.py --eval

```
```
CUDA_VISIBLE_DEVICES=0,1 python3 ./main.py --distributed --eval
```


# Citation
Please cite if you make use of the code.

```

@inproceedings{kim2026mclsv,
  title={Mixture Consistency Learning for Robust Speaker Verification in Noisy Environments},
  author={Seung-bin Kim and Chan-yeong Lim and Jungwoo Heo and Hyun-seo Shin and Kyo-Won Koo and Jisoo Son and Kyung-Wha Kim, and Ha-Jin Yu},
  booktitle={},
  year={2026}
}
```