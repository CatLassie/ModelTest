# Model Test
simple interface for sequence model testing

1: download and install miniconda from https://www.anaconda.com/download/success

2: create a conda environment from model_test.yml

    conda env create -n model_test -f model_test.yml

3: 

# conda environments cheat sheet:

create an environment from a .yml file:

conda env create -n my_env_name -f my_env_name.yml

export an environment as .yml file:

conda env export > my_env_name.yml

remove an environment

conda remove -n my_env_name --all