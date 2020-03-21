# moead-levy-python
A simple python script implmentation of MOEA/D injected with Levy Flight mutation

## Requirement

This program runs with ```Python 3```.

The following python libraries are required.
- numpy
- scipy
- yaml

## Usage

1. Set parameters for problem and algorithm in ```.yml``` file.

2. Run with the following command.

    ```python MOEAD.py [parameter yml file] [random seed]```

    Or use bash script in ```test.sh```

3. Result is saved as ```.csv``` files in ```./tmp``` folder by default.

## MOEA/D

An algorithm to solve multi-objective optimization problems (MOOPs). It decomposes the MOOP into several single-objective task and solve them together.

See the origin publication at:
- ZHANG, Qingfu; LI, Hui. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on evolutionary computation*, 2007, 11.6: 712-731.

## Levy Flight

A mutation method based on stable distribution (heavy-tailed). It can implement jump process with a small probability to escape from local optima. In my implementation, I use Levy Flight to enhance differential mutation as follows.

```y = xi + levy * (xi - xj)```

See the method of simulation of Levy random number at:
- MANTEGNA, Rosario N.; STANLEY, H. Eugene. Stochastic process with ultraslow convergence to a Gaussian: the truncated Lévy flight. *Physical Review Letters*, 1994, 73.22: 2946.
- GUTOWSKI, Marek. Levy flights as an underlying mechanism for global optimization algorithms. *arXiv preprint math-ph/0106003*, 2001.

##
