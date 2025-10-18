"""Group L1 proximal operators with various non-convex penalty functions.

This module implements group L1 proximal operators with different non-convex penalty functions
including Minimax Concave Penalty (MCP), Smoothly Clipped Absolute Deviation (SCAD),
and transformed L1 penalties for group sparse optimization.

The proximal operators are designed for structured sparsity problems where variables
are organized into groups, promoting group-wise sparsity patterns.

"""

from typing import Union, Any
import numpy as np

from src.prox.group_prox.general import *
from src.prox.sep_prox.prox_cl import ProxMCP, ProxSCAD, ProxTransformedl1

__all__ = ['L1_SubvecProx', 'L1_MCP_SubvecProx', 'L1_SCAD_SubvecProx', 'L1_Transformed']


class L1_MCP_SubvecProx(L1_SubvecProx):
    """Group L1 proximal operator with Minimax Concave Penalty (MCP).
    
    Implements the group L1 proximal operator with MCP penalty function, which provides
    nearly unbiased variable selection and maintains the convexity of the penalized loss
    for sufficiently large coefficients.
    
    The MCP penalty is defined as:
        ρ(t; λ, γ) = λ∫₀^|t| (1 - x/(γλ))₊ dx
    
    where (a)₊ = max(0, a) and γ > 1 is the concavity parameter.
    """
    
    def __init__(self, gLen: int, lamb: float, fix_param: float, precompute: bool):
        """
        Initialize the MCP group proximal operator.
        
        Args:
            gLen: Length of each group (number of variables per group)
            lamb: Regularization parameter λ controlling sparsity strength
            fix_param: Concavity parameter γ (must be > 1)
            precompute: Whether to precompute optimization conditions
        """
        self.element_prox = ProxMCP(fix_param)
        super().__init__(gLen, lamb, self.element_prox)
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        self.fix_param = fix_param
        # if lamb is not None and precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb: float) -> float:
        """Calculate condition 1 for MCP proximal operator.
        
        Condition 1 represents the threshold below which the entire group is set to zero.
        
        Args:
            lamb: Regularization parameter λ
            
        Returns:
            Threshold value for group sparsity
        """
        return lamb

    def calculate_con2(self, lamb: float) -> float:
        """Calculate condition 2 for MCP proximal operator.
        
        Condition 2 represents the threshold for individual coefficient shrinkage.
        
        Args:
            lamb: Regularization parameter λ
            
        Returns:
            Threshold value for coefficient-wise sparsity
        """
        return lamb

    def calculate_con4(self, lamb: float, s_arange: np.ndarray) -> np.ndarray:
        """Calculate condition 4 for MCP proximal operator.
        
        Condition 4 represents the piecewise linear threshold function
        that determines the shrinkage pattern for different group sizes.
        
        Args:
            lamb: Regularization parameter λ
            s_arange: Array of group sizes from 1 to gLen
            
        Returns:
            Array of threshold values for different group sizes
        """
        return np.where(lamb * s_arange < self.fix_param, lamb * s_arange, self.fix_param)

    def obj(self, y: np.ndarray, ytilde: np.ndarray, lamb: float) -> np.ndarray:
        """Calculate the objective function value for MCP penalty.
        
        The objective function combines the MCP penalty with the quadratic proximity term:
            J(y) = λ·ρ(||ỹ||₁; λ, γ) + 0.5·||y - ỹ||₂²
        
        where ρ is the MCP penalty function.
        
        Args:
            y: Original variable values
            ytilde: Transformed variable values after proximal operation
            lamb: Regularization parameter λ
            
        Returns:
            Objective function value
        """
        y_l1 = ytilde.sum()
        return lamb * np.where(y_l1 < self.fix_param, y_l1 - y_l1 ** 2 / (2 * self.fix_param),
                               self.fix_param / 2) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)


class L1_SCAD_SubvecProx(L1_SubvecProx):
    """Group L1 proximal operator with Smoothly Clipped Absolute Deviation (SCAD) penalty.
    
    Implements the group L1 proximal operator with SCAD penalty function, which provides
    oracle properties and maintains continuity in the thresholding rule.
    
    The SCAD penalty is defined as a symmetric function with derivative:
        ρ'(t) = λ·I(t ≤ λ) + [(aλ - t)₊/(a - 1)λ]·I(t > λ)
    
    for some a > 2, where I(·) is the indicator function.
    """
    precompute: bool

    def __init__(self, gLen: int, lamb: float, fix_param1: float, fix_param2: float, precompute: bool):
        """
        Initialize the SCAD group proximal operator.
        
        Args:
            gLen: Length of each group (number of variables per group)
            lamb: Regularization parameter λ controlling sparsity strength
            fix_param1: First tuning parameter a (must be > 2)
            fix_param2: Second tuning parameter (typically related to a)
            precompute: Whether to precompute optimization conditions
        """
        self.element_prox = ProxSCAD(fix_param1=fix_param1, fix_param2=fix_param2)
        super().__init__(gLen, lamb, self.element_prox)
        self.fix_param1 = fix_param1
        self.fix_param2 = fix_param2
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        # if lamb is not None and precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb: float) -> float:
        """Calculate condition 1 for SCAD proximal operator.
        
        Args:
            lamb: Regularization parameter λ
            
        Returns:
            Threshold value for group sparsity
        """
        return lamb * self.fix_param1

    def calculate_con2(self, lamb: float) -> float:
        """Calculate condition 2 for SCAD proximal operator.
        
        Args:
            lamb: Regularization parameter λ
            
        Returns:
            Threshold value for coefficient-wise sparsity
        """
        return lamb * self.fix_param1

    def calculate_con4(self, lamb: float, s_arange: np.ndarray) -> np.ndarray:
        """Calculate condition 4 for SCAD proximal operator.
        
        Args:
            lamb: Regularization parameter λ
            s_arange: Array of group sizes from 1 to gLen
            
        Returns:
            Array of threshold values for different group sizes
        """
        return np.where(lamb * s_arange < self.fix_param2,
                        lamb * self.fix_param1 * s_arange,
                        self.fix_param2 * self.fix_param1)

    def obj(self, y: np.ndarray, ytilde: np.ndarray, lamb: float) -> np.ndarray:
        """Calculate the objective function value for SCAD penalty.
        
        The objective function combines the SCAD penalty with the quadratic proximity term.
        
        Args:
            y: Original variable values
            ytilde: Transformed variable values after proximal operation
            lamb: Regularization parameter λ
            
        Returns:
            Objective function value
        """
        y_l1 = ytilde.sum()
        return lamb * np.where(
            y_l1 < self.fix_param1, self.fix_param1 * y_l1,
            np.where(y_l1 <= self.fix_param2 * self.fix_param1,
                     (2 * self.fix_param2 * self.fix_param1 * y_l1 - y_l1 ** 2 - self.fix_param1 ** 2) / (
                             2 * (self.fix_param2 - 1)),
                     (self.fix_param1 ** 2) * 0.5 * (self.fix_param2 + 1))) + 0.5 * (
                np.linalg.norm(y - ytilde, 2) ** 2)


class L1_Transformed(L1_SubvecProx):
    """Group L1 proximal operator with transformed L1 penalty.
    
    Implements the group L1 proximal operator with a transformed L1 penalty function,
    which provides a smooth transition between L1 and L0 penalties.
    
    The transformed L1 penalty is defined as:
        ρ(t) = (a + 1)·t / (a + t)
    
    where a > 0 is a tuning parameter that controls the degree of non-convexity.
    """
    
    def __init__(self, gLen: int, lamb: float, fix_param: float, precompute: bool):
        """
        Initialize the transformed L1 group proximal operator.
        
        Args:
            gLen: Length of each group (number of variables per group)
            lamb: Regularization parameter λ controlling sparsity strength
            fix_param: Tuning parameter a controlling non-convexity (must be > 0)
            precompute: Whether to precompute optimization conditions
        """
        self.element_prox = ProxTransformedl1(fix_param)
        super().__init__(gLen, lamb, self.element_prox)
        self.fix_param = fix_param
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        # if precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb: float, *args: Any, **kwargs: Any) -> Union[T, int, float, np.ndarray]:
        """Calculate condition 1 for transformed L1 proximal operator.
        
        Condition 1 represents the piecewise threshold function that determines
        the shrinkage pattern based on the regularization parameter.
        
        Args:
            lamb: Regularization parameter λ
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Threshold value for group sparsity
        """
        fix_param = self.fix_param
        return np.where(lamb <= 0.5 * fix_param ** 2 / (fix_param + 1),
                        lamb * (1 + 1 / fix_param),
                        1.5 * (2 * lamb * (fix_param + 1) * fix_param) ** (1 / 3) - fix_param)

    def calculate_con2(self, lamb: float, *args: Any, **kwargs: Any) -> Union[T, int, float, np.ndarray]:
        """Calculate condition 2 for transformed L1 proximal operator.
        
        Args:
            lamb: Regularization parameter λ
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Threshold value for coefficient-wise sparsity
        """
        fix_param = self.fix_param
        return np.where(lamb <= 0.5 * fix_param ** 2 / (fix_param + 1),
                        lamb * (1 + 1 / fix_param),
                        np.sqrt(2 * lamb * (fix_param + 1)) - 0.5 * fix_param)

    def calculate_con4(self, lamb: T, s_arange: np.ndarray) -> np.ndarray:
        """Calculate condition 4 for transformed L1 proximal operator.
        
        Args:
            lamb: Regularization parameter λ
            s_arange: Array of group sizes from 1 to gLen
            
        Returns:
            Array of threshold values for different group sizes
        """
        fix_param = self.fix_param

        return np.where(s_arange <= fix_param ** 2 / (2 * lamb * (1 + fix_param)),
                        s_arange * lamb * (1 + 1 / fix_param),
                        1.5 * (2 * s_arange * lamb * (fix_param + 1) * fix_param) ** (1 / 3) - fix_param)

    def obj(self, y: np.ndarray, ytilde: np.ndarray, lamb: T) -> np.ndarray:
        """Calculate the objective function value for transformed L1 penalty.
        
        The objective function combines the transformed L1 penalty with the quadratic proximity term.
        
        Args:
            y: Original variable values
            ytilde: Transformed variable values after proximal operation
            lamb: Regularization parameter λ
            
        Returns:
            Objective function value
        """
        y_l1 = ytilde.sum()
        return lamb * (self.element_prox.a + 1) * y_l1 / (self.element_prox.a + y_l1) + 0.5 * (
                (y - ytilde) ** 2).sum()  # np.linalg.norm(y - ytilde, 2) ** 2
