# Collection of Conservation Laws Solver in 1D

This repository provides a framework for solving Conservation Law problems in 1D
with finite element and finite volume methods. Execute the program with `cls1`
(Conservation Laws Solver 1D).

The program uses `Python 3.10.6`. We suggest to use a virtual environment. The
needed packages can be installed via:

    pip install -r requirements.txt

After installation feel free to inspect the program by typing:

    ./cls1 -h

Due to storage issues, the data-driven solvers must be generated.
Use the commands `scripts/gen_solver_1` and `scripts/gen_solver_2` for this. Note, this may take some time.

The results of the authors thesis can be reproduced via `scripts/gen_results` and are stored in in the `data/media` directory.
