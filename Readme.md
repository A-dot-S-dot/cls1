# Collection of Conservation Laws Solver in 1D

This repository provides a framework for solving Conservation Law problems in 1D
with finite element, finite volume and machine learning methods. Execute the 
program with `cls1` (Conservation Laws Solver 1D).

The program uses `Python 3.10.6`. We suggest to use a virtual environment. For
that simply run

    python -m venv venv

and 

    source venv/bin/activate

to activate the virtual enviornment. The needed packages can be installed via:

    pip install -r requirements.txt

After installation feel free to inspect the program by typing:

    ./cls1 -h

The data-driven solver were generated with `scripts/gen_solver_1` and
`scripts/gen_solver_2`.

The results of the authors thesis can be reproduced via `scripts/gen_results`
and are stored in the `data/media` directory.
