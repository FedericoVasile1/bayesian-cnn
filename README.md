# Uncertainty Modelling with CNNs: A Comparison of Implementations

## Introduction 

The aim of this project is to compare a network implemented via Variational Inference with the same network implemented via Monte Carlo Dropout.
Extensive experiments were carried out on MNIST, CIFAR10, COLORECTAL CANCER (CRC).

## Overview

The project is organised as follows:

    ├── data/                           {contains the datasets samples}
    │   ├── ...
    │   ├── data_info.json              {contains general info about datasets}
    ├── lib/                            {contains external libraries/repositories}
    |   ├── ...
    ├── src/
    |   ├── datasets/                   {contains PyTorch Dataset modules to load data}
    |   |   ├── *.py
    |   ├── models/                     {contains the PyTorch Models}
    |   |   ├── mc_dropout/
    |   |   |   ├── *.py
    |   |   ├── var_inf/
    |   |   |   ├── *.py
    |   ├── tools/                      {contains training and evaluation pipelines}
    |   |   ├── mc_dropout/
    |   |   |   ├── *.py
    |   |   ├── var_inf/
    |   |   |   ├── *.py
    |   ├── utils/                      {contains general .py modules organised by purpose}
    |   |   ├── *.py
    |   |   ├── colab_notebooks/        {contains useful notebooks to be imported into
    |   |   |   ├── *.ipynb              Colab for automatic installation and running}


## Environment

The code is developed with ***Python 3.7***, ***PyTorch 1.4***.

Regarding the network with Variational Inference, the implementation provided by [Shridhar et al.](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) was used, please refer to it for a full description. The repository will be imported under `lib/` folder and used during the project as off-the-shelf framework in oder to build and run the Variational Inference model.


## Installation
  ### Opt. 1: Colab automatic installation and running (*recommended*):
  The folder `src/utils/colab_notebooks` contains some useful notebooks designed to be imported in Google Colab with all commands ready to go.<br><br>
  **Main notebook**:
  * `run_Project8.ipynb`: contains all installation steps and an example run for this repository.
  
  Other useful notebooks:
  * `run_PyTorch_BayesianCNN.ipynb`: contains all installation steps and an example run for the only [external repository](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)  about Variational Inference implementation.
  * `visualizing_crc.ipynb`: contains simple code to visualize random CRC samples.
      
  ### Opt. 2: Manual installation:
  NOTE: it is recommended to create a Python 3.7 virtual environment.

  1. Clone repository: 
      ```
      git clone https://github.com/FedericoVasile1/Project8
      cd Project8
      ```

  2. Clone external repository:
      ```
      git submodule init
      git submodule update    
      ```

  3. Install dependencies:
      ```
      pip install requirements.txt    
      ```

  4. Download CRC dataset. It will be located at `data/crc_3_noisy/`.
      ```
      bash download_crc_3_noisy.sh
      ```

  5. Convert CRC labels from char to int. Labels file will be: `data/crc_3_noisy/real_classes_{train|test}.npy`.
      ```
      python src/utils/labels_conversion.py
      ```

  5.  Create folders for Tensorboard:
      ```
      mkdir results
      mkdir results/var_inf
      mkdir results/mc_dropout
      ```
    
## Training
  * Monte Carlo Dropout model:
  
    ```
    python src/tools/mc_dropout/train.py --dataset CRC\
                                         --dropout 0.1\
                                         --activation_function softplus\
                                         --batch_size 256\
                                         --epochs 100
    ```
    NOTE: for the entire list of arguments check `src/tools/mc_dropout/train.py` at the bottom of the module.

  * Variational Inference model:
  
    ```
    !python src/tools/var_inf/train.py --dataset CRC\
                                       --activation_function relu\
                                       --batch_size 256\
                                       --epochs 100\
                                       --prior_mu 0.0\
                                       --prior_sigma 0.1
    ```
    NOTE: for the entire list of arguments check `src/tools/var_inf/train.py` at the bottom of the module.
  
## Evaluation
  * Monte Carlo Dropout model:
  
    ```
    python src/tools/mc_dropout/eval.py --dataset CRC\
                                         --dropout 0.1\
                                         --activation_function softplus\
                                         --batch_size 256\
                                         --K 100
    ```
    NOTE: for the entire list of arguments check `src/tools/mc_dropout/eval.py` at the bottom of the module.

  * Variational Inference model:
  
    ```
    !python src/tools/var_inf/eval.py --dataset CRC\
                                       --activation_function relu\
                                       --batch_size 256\
                                       --prior_mu 0.0\
                                       --prior_sigma 0.1\
                                       --K 100
    ```
    NOTE: for the entire list of arguments check `src/tools/var_inf/eval.py` at the bottom of the module.

## Documents
  * [report](https://drive.google.com/file/d/1TvDcZxhbcPg9HKvgo1L_UbIG-bJ-xgWa/view?usp=sharing)
  * [presentation](https://drive.google.com/file/d/1ZAITcbu1DA3YtJh4-v6cAdHTkViuRxvi/view?usp=sharing)

## Acknowledgements
  * [Shridhar et al., 2019, A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference](https://arxiv.org/pdf/1901.02731.pdf)
  * [Gal et al., 2016, Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)  
  
