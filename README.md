# 540 Project Details

## How to run code on MNIST TLDR
1. Create a new virtual environment for the project using `venv` or `conda` (optional but recommended)
2. Install a Fortrain compiler like `gfortran`
3. `cd` to this directory
4. Run `pip install -r requirements.txt --no-cache-dir` (replace tensorflow-gpu with tensorflow if your machine doesn't have a gpu)
5. Download MNIST dataset from [here](https://www.kaggle.com/avnishnish/mnist-original) and unzip to `datasets` directory
6. Copy the file `dependencies/exp_kern.py` to `<PYTHON PATH>/lib/<PYTHON VERSION>/site-packages/GPy/kern/src`
7. Copy the contents of the file `dependencies/scipydirect_for_bo_bos.py` and replace the contents of the file `<PYTHON PATH>/lib/<PYTHON VERSION>/site-packages/scipydirect/__init__.py` with the copied contents.
8. Add the line `from .src.exp_kern import ExpKernel` to `<PYTHON PATH>/lib/<PYTHON VERSION>/site-packages/GPy/kern/__init__.py`
9. Add the line `from .exp_kern import ExpKernel` to `<PYTHON PATH>/lib/<PYTHON VERSION>/site-packages/GPy/kern/src/__init__.py`
10. Run the code via `python run_bo_bos.py`


## Project Todo
1. (1 person) Modify the `run_BOS` function at `bos_function.py` by giving the BOS GP an option to accept not just the loss curve but also extra information that comes with the loss. Preprocess the extra information appropriately by scaling etc.
2. (2 people) Add a function per hyperparameter optimization task at `objective_functions.py`, specifying the hyperparameter tuning range, training logic, evaluation logic, etc.


# Original README

Code for the following paper:

Zhongxiang Dai, Haibin Yu, Kian Hsiang Low and Patrick Jaillet. "Bayesian Optimization
Meets Bayesian Optimal Stopping." In International Conference on Machine Learning (ICML),
Long Beach, CA, Jun 9-15, 2019.



Description of the scripts:
* bayesian_optimization.py: the BO algorithm; implements both standard GP-UCB and BO-BOS
* helper_funcs.py: some helper functions (e.g. acquisition functions) for the BO algorithm
* bos_function.py: contains the Bayesian optimal stopping algorithm
* objective_functions.py: contains several objective functions for hyper-parameter tuning
* run_bo_bos.py: the wrapper script which calls the BO-BOS algorithm
* analyze_results.ipynb: an ipython notebook script analyzing the results obtained by running the "run_bo_bos.py" script (assuming "objective_function_LR_MNIST" is used as the objective function)
* generate_mnist_training_validation.py: generate the training set/validation set split for the MNIST dataset


Description of the directories:
* datasets: contains the datasets used for hyper-parameter tuning
    MNIST: please download the "mnist-original.zip" from "https://www.kaggle.com/avnishnish/mnist-original", and unzip to folder "datasets/";
        then, run "generate_mnist_training_validation.py", which will generate the training set/validation set split.
    SVHN: please download the files "train.tar.gz" and "test.tar.gz" from "http://ufldl.stanford.edu/housenumbers/", 
        and put then in the folder "datasets/"
    CIFAR-10: this dataset will be automatically downloaded by the keras package
* dependencies: contains some dependency scripts, which are explained in more detail below
* saved_init: contains the initializations used by the BO/BO-BOS algorithm; since we would like to use the same initializations for both GP-UCB and BO-BOS
* saved_results: contains the results of the BO/BO-BOS algorithm; the results are saved/updated after every iteration


key dependencies (excluding commonly used packages such as scipy, numpy, tensorflow, keras, etc.)
* GPy
    * install GPy
    * add the line "from .src.exp_kern import ExpKernel" to "PYTHON_PATH/lib/python3.5/site-packages/GPy/kern/\_\_init\_\_.py"
    * add the line "from .exp_kern import ExpKernel" to "PYTHON_PATH/lib/python3.5/site-packages/GPy/kern/src/\_\_init\_\_.py"
    * place the script "exp_kern.py" in the "dependencies" folder to the folder "PYTHON_PATH/lib/python3.5/site-packages/GPy/kern/src/"
* scipydirect: this package uses the DIRECT method to optimize the acquisition function
    * install scipydirect with "pip install scipydirect"
    * replace the content of the script "PYTHON_PATH/lib/python3.5/site-packages/scipydirect/\_\_init\_\_.py" with the content of the script "scipydirect_for_bo_bos.py" in the "dependencies" folder; this step is required since we modified the interface of the scipydirect minimize function
