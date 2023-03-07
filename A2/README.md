# ELE6310E Assignment 2

First, read all of the assignment questions in the PDF and, then, follow the instructions in `main.ipynb` using Google Colab.

1- Clone this github repo and upload the contents of first assignment to your drive. You will be prompted to allow access to your drive folder.

2- In the section entitled "Link your assignment folder & install requirements", please write the full path to the directory you will be working in. If you are in your root, this would be: `/content/gdrive/MyDrive/`.

3- Please follow the instructions in the notebook very carefully. The docstring in `solution.py` should also be helpful.

## Install timeloop-Accelergy
```
git clone https://github.com/kmchiti/ELE6310E.git
source ELE6310E/A2/install_timeloop/install_timeloop.sh
# (if necessary)
export PATH=$PATH:~/.local/bin
# make sure timeloop executables can be found
which -a timeloop-model
# make sure accelergy executable can be found
which -a accelergy
```

# Question 1
You can run Accelergy to get the energy consumption using following command:
```
timeloop-model Q1/arch/*.yaml  Q1/prob/*.yaml  Q1/map/Q1_ws.map.yaml
```
You can extract the energy consumption, memory accesses, and all other stats from the `timeloop-model.stats.txt`. 

# Question 2
Use `YAML_generator` in`utils.py` to generate the YAML files for the different architecture configurations. Rather than using timeloop-metrics, the energy reference table (ERT) can then be generated with following command:
```
timeloop-model Q2/arch.yaml Q2/prob.yaml Q2/map.yaml
```
In this command, the `prob.yaml` and `map.yaml` files are not important. Default ones are provided.
Then, you can use `YAML_parser` to parse the ERT output `timeloop-model.ERT.yaml` and retrieve the energy for read and write operations.

# Question 3
First, complete the `model_to_spars` and `generate_resnet_layers` in `solution.py`. Then follow instructions in `main.ipynb` notebook and prune the network. After fine-tuning, save the model and generate the YAML files for each layers of the pruned network. Then you can use `run_Accelergy` to estimate the energy consumption of pruned network.
