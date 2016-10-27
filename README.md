## Installation

To create a Conda environment with name {ENV_NAME}, run
`conda env create -n {ENV_NAME} -f environment.yml`

## Running

To test a very basic example, run
`python exp.py -c config/config_std_tiny.json -p sample_policies.json`.

This should create `config_std_tiny.json_sample_policies.json.csv` in the main directory.
