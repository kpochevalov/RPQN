import numpy as np

class BaseProblem:
    def __init__(self):
        pass

    def grad_f(self, x: np.ndarray):
        """
        Compute the gradient of the function f at point x.
        :param x: Input point.
        :return: Gradient of f at x.
        """
        raise NotImplementedError()

    def prox(self, x: np.ndarray, gamma: float):
        """
        Compute the proximal operator of the function \phi at point x
        with respect to matrix (\gamma I).
        :param x: Input point.
        :param gamma: Scaling factor.
        :return: Proximal operator of \phi at x.
        """
        raise NotImplementedError()

    def prox_jacob(self, x: np.ndarray, gamma: float, C: np.ndarray):
        """
        Compute the Jacobian of the proximal operator of the function \phi
        at point x with respect to matrix (\gamma I) multiplied by matrix C on the right.
        :param x: Input point.
        :param gamma: Scaling factor.
        :param C: Matrix to multiply on the right.
        :return: Jacobian of the proximal operator of \phi at x.
        """
        raise NotImplementedError()

    def phi(self, x: np.ndarray):
        """
        Compute the value of the function \phi at point x.
        :param x: Input point.
        :return: Value of \phi at x.
        """
        raise NotImplementedError()

    def psi(self, x: np.ndarray):
        """
        Compute the value of the function \psi at point x.
        :param x: Input point.
        :return: Value of \psi at x.
        """
        raise NotImplementedError()

    def crit(self, x):
        """
        Compute the norm of (\prox(x - \\nabla f(x), 1) - x).
        :param x: Input point.
        :return: Norm of the difference.
        """
        raise NotImplementedError()