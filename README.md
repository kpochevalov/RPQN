# RPQN: Regularized Proximal Quasi-Newton Method

This repository contains a Python implementation of the **Regularized Proximal Quasi-Newton (RPQN)** method as described in [Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems](https://arxiv.org/abs/2210.07644).

## Features

- Modular design with a base problem class for easy extension.
- Implements several quasi-Newton update strategies (BFGS, SR1, PSB, DFP).
- Example usage in a Jupyter Notebook for solving regularized least squares problems.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── RPQN.ipynb
└── src/
    ├── base_problem.py
    ├── config.py
    └── rpqn.py
```

## Getting Started

### 1. Install Requirements

Install the required Python packages:

```sh
pip install -r requirements.txt
```

### 2. Running the Example in Jupyter Notebook

1. Open [`RPQN.ipynb`](RPQN.ipynb) in Jupyter Notebook or Visual Studio Code.
2. Run the notebook cells sequentially. The notebook demonstrates:
    - How to define a problem by subclassing [`src.base_problem.BaseProblem`](src/base_problem.py).
    - How to configure and run the RPQN algorithm using [`src.rpqn.RPQN`](src/rpqn.py) and [`src.config.RPQN_config`](src/config.py).
    - How to visualize convergence and residuals.

#### Example Workflow

- **Define your problem**:  
  The notebook provides an example `MSE_L2_problem` class for L2-regularized least squares.
- **Configure RPQN**:  
  Adjust parameters in [`RPQN_config`](src/config.py) as needed.
- **Run the algorithm**:  
  Call [`RPQN(config, problem, x0, x_true)`](src/rpqn.py) as shown in the notebook.
- **Visualize results**:  
  Use matplotlib to plot convergence metrics.

## Customization

To solve your own problem:
1. Subclass [`BaseProblem`](src/base_problem.py) and implement the required methods.
2. Follow the usage pattern in [`RPQN.ipynb`](RPQN.ipynb).

## References

- [Regularized Proximal Quasi-Newton Methods for Convex Composite Optimization](https://arxiv.org/abs/2210.07644)

---

For questions or contributions, please open an issue or pull request.
