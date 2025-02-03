# Model Test
simple interface for sequence model testing

1: download and install miniconda from https://www.anaconda.com/download/success

2: create a conda environment from model_test.yml

    conda env create -n model_test -f model_test.yml

create conda environment with 

3: activate the environment by:

    conda activate model_test

3: save a model locally by running the save_model.py script

    python save_model.py <name_of_the_model>

e.g. python save_model.py best-base-cased

4: run the model:

    python main.py <model_name> <test_text_sequence>

e.g. python main.py best-base-cased "hello there!"

# conda environments cheat sheet:

create an environment from a .yml file:

conda env create -n my_env_name -f my_env_name.yml

export an environment as .yml file:

conda env export > my_env_name.yml

remove an environment

conda remove -n my_env_name --all

# manually creating conda environment

conda env create -n model_test python conda pip

activate environment

install torch with pip, e.g. 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

(see https://pytorch.org/get-started/locally/)

install the hugging face transformers package:

pip install transformers