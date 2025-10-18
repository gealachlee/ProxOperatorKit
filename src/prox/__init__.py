from abc import ABCMeta, abstractmethod
from typing import Optional, Any

import numpy as np

__all__ = ['ProximalOperator', 'GroupProximalOperator']


class ProximalOperator(metaclass=ABCMeta):
    """Abstract base class for proximal operators.
    
    Proximal operators are fundamental components in optimization algorithms,
    particularly in sparse optimization and convex optimization problems.
    """
    
    latex_name: Optional[str] = None
    """LaTeX representation of the proximal operator for mathematical notation."""

    def calculate_subderivative(self, u: np.ndarray[Any, Any], *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Calculate the subderivative of the proximal operator at point u.
        
        Args:
            u: Input point where subderivative is evaluated
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Subderivative value at point u
        """
        raise NotImplementedError

    @classmethod
    def obj(cls, x: np.ndarray[Any, Any], xtilde: np.ndarray[Any, Any], nu: float) -> np.ndarray[Any, Any]:
        """Calculate the objective function value.
        
        Args:
            x: Current variable value
            xtilde: Reference variable value
            nu: Regularization parameter
            
        Returns:
            Objective function value
        """
        raise NotImplementedError

    @classmethod
    def prox(cls, x: np.ndarray[Any, Any], v: float, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Calculate the proximal operator.
        
        The proximal operator is defined as:
            prox_{λf}(x) = argmin_y {f(y) + (1/2λ)||y - x||^2}
        
        Args:
            x: Input vector or matrix
            v: Threshold parameter (typically λ)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Result of the proximal operator application
        """
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        """Get the name of the proximal operator.
        
        Returns:
            Class name as string
        """
        return cls.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make the instance callable, equivalent to calling prox().
        
        Returns:
            Result of the proximal operator
        """
        return self.prox(*args, **kwargs)

    def __str__(self) -> str:
        """String representation of the proximal operator.
        
        Returns:
            String representation
        """
        return self.name()


class GroupProximalOperator(ProximalOperator, metaclass=ABCMeta):
    """Abstract base class for group-structured proximal operators.
    
    Group proximal operators handle structured sparsity where variables
    are organized into groups, promoting group-wise sparsity patterns.
    """
    
    n: int
    """Dimension of the input vector."""
    
    gLen: int
    """Length of each group."""
    
    num_subvectors: int
    """Number of subvectors (groups) in the input."""
    
    latex_name: Optional[str] = None
    """LaTeX representation for mathematical notation."""

    __slots__ = ('n', 'gLen', 'num_subvectors', 'num_samples')

    @staticmethod
    def get_num_subvectors(n: int, gLen: int) -> int:
        """Calculate the number of subvectors given dimension and group length.
        
        Args:
            n: Total dimension
            gLen: Length of each group
            
        Returns:
            Number of subvectors (n // gLen)
        """
        return n // gLen

    @classmethod
    def obj(cls, x: np.ndarray[Any, Any], xtilde: np.ndarray[Any, Any], nu: float) -> np.ndarray[Any, Any]:
        """Calculate the group-sparse objective function.
        
        The objective function typically has the form:
            J(x) = ν * ||x||_group + 0.5 * ||x - x̃||²
        
        Args:
            x: Current variable value
            xtilde: Reference variable value
            nu: Regularization parameter
            
        Returns:
            Objective function value
        """
        raise NotImplementedError

    def _split_vectors(self, x: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
        """Split the input vector into group subvectors.
        
        Args:
            x: Input vector to split
            
        Returns:
            List of subvectors, each representing a group
        """
        raise NotImplementedError

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Concatenate group subvectors back into a single vector.
        
        Args:
            x_subvectors: List of group subvectors
            
        Returns:
            Concatenated vector
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make the instance callable, equivalent to calling prox().
        
        Returns:
            Result of the group proximal operator
        """
        return super().__call__(*args, **kwargs)
