#!/bin/bash

#SBATCH -J hello               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 9                 # Total number of nodes requested
#SBATCH -n 144                 # Total number of mpi tasks requested
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours

# Launch MPI-based executable
prun ./hello
