#!/bin/bash

#SBATCH --job-name perplexity
#SBATCH -N 2
#SBATCH -p broadwell
# Use modules to set the software environment

python expts/expts_v5.py
