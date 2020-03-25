# DetGP
Code for the AAAI 2020 paper: Dynamic Embedding on Textual Networks via a Gaussian Process.


This repository contains source code necessary to reproduce the results presented in the following paper:
* [*Dynamic Embedding on Textual Networks via a Gaussian Process*](https://arxiv.org/abs/1910.02187) (AAAI 2020)

This project is maintained by [Pengyu Cheng](https://linear95.github.io/). Feel free to contact pengyu.cheng@duke.edu for any relevant issues.

## Dependencies: 
This code is written in python. The dependencies are:
* Python 2.7
* Tensorflow>=1.13 (1.13.1 is recommended)

## Download Data
Download the folder `./datasets/` from the [CANE](https://github.com/thunlp/CANE) repository, which includes *Cora* and *HepTh* datasets.

## Run the code
To reproduce the link prediction results of DetGP, run the command:
```
python train_link_predict.py
```

To reproduce the node classification results of DetGP, run the command:
```
python train_node_classify.py
```

The parameters for experiments can be set in `config.py`, *e.g.*, the choice of dataset, the train-test split ratios.

## Citation 
Please cite our AAAI paper if you found the code useful.

```latex
@misc{cheng2019dynamic,
    title={Dynamic Embedding on Textual Networks via a Gaussian Process},
    author={Pengyu Cheng and Yitong Li and Xinyuan Zhang and Liqun Cheng and David Carlson and Lawrence Carin},
    year={2019},
    eprint={1910.02187},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

