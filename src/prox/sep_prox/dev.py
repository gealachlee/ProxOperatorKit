from pyhypergeomatrix.hypergeomat import hypergeomPQ
import numpy as np
import matplotlib.pyplot as plt
from prox import ProximalOperator


# def hypergeom1over2(y,Lambda):
#     epsilon=(1/2)*Lambda*np.power(y,-3/2)
#     t=(27/4)*epsilon**2
#     return 2/3+(1/3)*hypergeomPQ(10,[-1/3,1/3],[1/2],[t])-epsilon*hypergeomPQ(10,[1/6,5/6],[3/2],[t]) #Gauss hypergeometric function
# def hypergeom2over3(y,Lambda):
#     epsilon=(2/3)*Lambda*np.power(y,-4/3)
#     t=(256/27)*epsilon**3
#     r=3/4+(1/4)*hypergeomPQ(10,[1/2,-1/4,1/4],[1/3,2/3],[t],2)
#     r=r-epsilon*hypergeomPQ(10,[1/12,7/12,5/6],[2/3,4/3],[t],2)-(epsilon**2/3)*hypergeomPQ(10,[5/12,11/12,7/6],[4/3,5/3],[t],2)
#     return r
# def hypergeom1over3(y,Lambda):
#     epsilon=(1/3)*Lambda*np.power(y,-5/3)
#     t=(3125/54)*epsilon**3
#     r=(4/5+(1/5)*hypergeomPQ(10,[-1/5,1/5,2/5,3/5],[1/3,1/2,2/3],[t],3)
#        -
#        epsilon*hypergeomPQ(10,[2/15,8/15,11/15,14/15],[2/3,5/6,4/3],[t],3))
#     r=r-((2/3)*epsilon**2)*hypergeomPQ(10,[4/15,7/15,13/15,16/15],[7/6,4/3,5/3],[t],3)
#     return r
#
# Lambda=1
# print('Lambda', Lambda)
# tau=np.linspace(0, 3, 500)
# ProxL1over2=np.zeros(np.shape(tau)[0])
# ProxL2over3=np.zeros(np.shape(tau)[0])
# ProxL1over3=np.zeros(np.shape(tau)[0])
#
# #proximal operator of L1/2
# threshold_1over2=1.5*np.power(Lambda,2/3)
# print('threshold 1over2= ',threshold_1over2)
# for i in range(np.shape(tau)[0]):
#     if tau[i]<=threshold_1over2:
#         ProxL1over2[i]=0
#     else:
#         ProxL1over2[i]=tau[i]*hypergeom1over2(tau[i],Lambda)
#
# #proximal operator of L2/3
# threshold_2over3=2*np.power(2*Lambda/3,3/4)
# print('threshold_2over3',threshold_2over3)
# for i in range(np.shape(tau)[0]):
#     if tau[i]<=threshold_2over3:
#         ProxL2over3[i]=0
#     else:
#         ProxL2over3[i]=tau[i]*hypergeom2over3(tau[i],Lambda)
#
class Prox1over3(ProximalOperator):

    @classmethod
    def _cal_hypergeom1over3(cls, y, Lambda):

        epsilon = (1 / 3) * Lambda * np.power(y, -5 / 3)

        t = (3125 / 54) * (epsilon ** 3)
        t = np.where(t>10,(5/4)*(4 *Lambda / 3)**(3/5),t)-1e-5
        r = 4 / 5 + (1 / 5) * hypergeomPQ(5, [-1 / 5, 1 / 5, 2 / 5, 3 / 5], [1 / 3, 1 / 2, 2 / 3], t,
                                          3) - epsilon * hypergeomPQ(5, [2 / 15, 8 / 15, 11 / 15, 14 / 15],
                                                                     [2 / 3, 5 / 6, 4 / 3], t, 3)
        - ((2 / 3) * (epsilon ** 2)) * hypergeomPQ(5, [4 / 15, 7 / 15, 13 / 15, 16 / 15], [7 / 6, 4 / 3, 5 / 3],
                                                         t, 3)
        r=np.where(abs(r)>=1,0,r)
        return r

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        condition = (5/4)*(4 *v / 3)**(3/5)#np.power(,  3 / 5)
        absx = np.abs(x)
        #     return np.where(absx <= condition, 0.0, x*np.array(
        #         [cls._cal_hypergeom1over3(abs(i), v) for i in  x.reshape(-1)])
        #
        # )
      #  return np.where(absx <= condition, 0.0, x * cls._cal_hypergeom1over3(absx, v))

        return  np.where(absx <= condition,0,x*np.abs(cls._cal_hypergeom1over3(absx, v)))

    @classmethod
    def name(cls):
        return 'L1over3'

    @classmethod
    def latex_name(cls):
        return r'$\ell_{1/3}$'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class Prox1over2(ProximalOperator):
    @classmethod
    def _cal_hypergeom1over2(cls, y, Lambda):
        epsilon = (1 / 2) * Lambda * np.power(y, -3 / 2)
        t = (27 / 4) * epsilon ** 2
        return 2 / 3 + (1 / 3) * hypergeomPQ(10, [-1 / 3, 1 / 3], [1 / 2], [t]) - epsilon * hypergeomPQ(10,
                                                                                                        [1 / 6, 5 / 6],
                                                                                                        [3 / 2], [
                                                                                                            t])  # Gauss hypergeometric function

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        condition = 1.5 * np.power(v, 2 / 3)
        absx = np.abs(x)
        return np.where(absx <= condition, 0, x * cls._cal_hypergeom1over2(absx, v))

    @classmethod
    def name(cls):
        return 'L1over2'

    @classmethod
    def latex_name(cls):
        return r'$\ell_{1/2}$'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()
