# CodeErrorDetection
Investigating the use of a DL system for detecting errors in code. 

<!-- vim-markdown-toc -->

## Getting Started

#### Install Miniconda

Please follow these [instructions](https://docs.conda.io/en/latest/miniconda.html) to install Miniconda (Python 3.8).

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Create and Activate a Conda Environment

The Conda environment used for the tutorial is specified in the `environment.yml` file. Creating the environment is a _on-time_ operation:
```
cd CodeErrorDetection
conda env create -f environment.yml
```

You should close the terminal and open a new one to make sure that Conda is correctly setup in your environment.

In any new console, remember to activate the newly created environemnt:
```
conda activate codeErrorDetection
```

[Here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) you can find more instructions on how to create and manage a Conda environment.

#### Run Data Generation

```
python generateData <filename.py>
```
