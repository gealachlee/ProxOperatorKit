"""Base model classes for optimization algorithms.

This module provides abstract base classes for various optimization models
used in sparse optimization problems, including group sparse and joint sparse models.
"""

from abc import ABCMeta, abstractmethod
from numpy import floating
from typing import Any, Optional

import numpy as np

__all__ = ['Model', 'JointModel', 'GroupModel']

from prox import GroupProximalOperator, ProximalOperator


class Model(metaclass=ABCMeta):
    """Abstract base class for optimization models.
    
    Models implement optimization algorithms that solve sparse optimization problems
    using proximal operators and iterative methods.
    """
    
    iter_history: list[np.ndarray[Any, Any]] = []
    """History of iterations for convergence analysis."""

    def __init__(self, A: np.ndarray[Any, Any], tau: float, **kwargs: Any) -> None:
        """Initialize the optimization model.
        
        Args:
            A: Observation matrix or sensing matrix
            tau: Regularization parameter for the optimization problem
            **kwargs: Additional model-specific parameters
        """
        self.A = A
        """Observation matrix."""
        
        self.tau = tau
        """Regularization parameter."""
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Perform one forward iteration of the optimization algorithm.
        
        Returns:
            Updated variable value after one iteration
        """
        raise NotImplementedError

    @abstractmethod
    def desc(self) -> str:
        """Get descriptive string about the model and its proximal operator.
        
        Returns:
            Descriptive string
        """
        raise NotImplementedError

    @abstractmethod
    def proximal_operator_name(self) -> str:
        """Get the name of the proximal operator used by this model.
        
        Returns:
            Proximal operator name
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make the model callable, equivalent to calling forward().
        
        Returns:
            Result of the forward pass
        """
        return self.forward(*args, **kwargs)


class GroupModel(Model):
    """Base class for group sparse optimization models.
    
    Group models handle optimization problems with group-structured sparsity,
    where sparsity is enforced at the group level rather than individual elements.
    """
    
    def __init__(self, A: np.ndarray, tau: float, prox_func: GroupProximalOperator) -> None:
        """Initialize group sparse model.
        
        Args:
            A: Sensing matrix
            tau: Regularization parameter
            prox_func: Group proximal operator for the optimization
        """
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        """Dimensions of the sensing matrix."""
        
        self.L_np: floating[Any] = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        """Lipschitz constant of the quadratic term."""
        
        self.gamma = 1 / np.linalg.norm(A, 2) ** 2
        """Step size parameter."""
        
        self.prox_func: GroupProximalOperator = prox_func
        """Group proximal operator used in the optimization."""
        
        self.iter_history: list[np.ndarray[Any, Any]] = []
        """History of iterations."""

    def desc(self) -> str:
        """Get description of the model and its proximal operator.
        
        Returns:
            Descriptive string
        """
        return f'Algorithm: {self.__class__.__name__} with proximal operator: {self.prox_func.name()}'

    def proximal_operator_name(self) -> str:
        """Get the name of the proximal operator.
        
        Returns:
            Proximal operator name
        """
        return self.prox_func.name()

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make the model callable, equivalent to calling forward().
        
        Returns:
            Result of the forward pass
        """
        return self.forward(*args, **kwargs)


class JointModel(Model):
    """Base class for joint sparse optimization models.
    
    Joint models handle optimization problems with both element-wise and group-wise
    sparsity constraints simultaneously.
    """
    
    def __init__(self, A: np.ndarray, tau: float, 
                 prox_func1: ProximalOperator, prox_func2: GroupProximalOperator) -> None:
        """Initialize joint sparse model.
        
        Args:
            A: Sensing matrix
            tau: Regularization parameter
            prox_func1: Element-wise proximal operator
            prox_func2: Group proximal operator
        """
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        """Dimensions of the sensing matrix."""
        
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        """Lipschitz constant of the quadratic term."""
        
        self.prox_func1: ProximalOperator = prox_func1
        """Element-wise proximal operator."""
        
        self.prox_func2: GroupProximalOperator = prox_func2
        """Group proximal operator."""
        
        self.iter_history: list[np.ndarray[Any, Any]] = []
        """History of iterations."""

    def desc(self) -> str:
        """Get description of the model and its proximal operators.
        
        Returns:
            Descriptive string
        """
        return f'Algorithm: {self.__class__.__name__} with proximal operators {self.prox_func1.name()} and {self.prox_func2.name()}'

    def proximal_operator_name(self) -> str:
        """Get the names of both proximal operators.
        
        Returns:
            Combined proximal operator names
        """
        return f"{self.prox_func1.name()} and {self.prox_func2.name()}"

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make the model callable, equivalent to calling forward().
        
        Returns:
            Result of the forward pass
        """
        return self.forward(*args, **kwargs)
