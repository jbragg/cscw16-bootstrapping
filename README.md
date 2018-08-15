## About

This code produces the decision-theoretic bootstrapping experiments from the following paper:
- Shih-Wen Huang, Jonathan Bragg, Isaac Cowhey, Oren Etzioni, Daniel S. Weld. [Toward Automatic Bootstrapping of Online Communities Using Decision-theoretic Optimization](https://www.cs.washington.edu/ai/pubs/huang-cscw16.pdf). In Proceedings of the 19th ACM Conference on Computer-Supported Cooperative Work & Social Computing (CSCW '16). 2016.

## Installation

To create a Conda environment with name {ENV_NAME}, run
`conda env create -n {ENV_NAME} -f environment.yml`

## Running

To test a very basic example, run
`python exp.py -c config/config_std_tiny.json -p sample_policies.json`.

This should create `config_std_tiny.json_sample_policies.json.csv` in the main directory.
