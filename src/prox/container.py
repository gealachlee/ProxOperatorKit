from prox.group_prox import ProxL1_1over2, ProxL1_2over3
from prox.group_prox.l1psi import *
from prox.group_prox.l2psi import *
from prox.sep_prox.prox_cl import *

from dependency_injector import containers, providers


class ProximalContainer(containers.DeclarativeContainer):
    n: providers.Dependency[int] = providers.Dependency(instance_of=int)
    gLen: providers.Dependency[int] = providers.Dependency(instance_of=int)
    data_size: providers.Dependency[int] = providers.Dependency(instance_of=int)

    precompute: providers.Dependency[bool] = providers.Dependency(instance_of=bool, default=False)

    # Element wise proximal operators factory
    prox_l0 = providers.Factory(ProxL0)
    prox_l1 = providers.Factory(ProxL1)
    prox_l1over2 = providers.Factory(ProxL1over2)
    prox_l2over3 = providers.Factory(ProxL2over3)
    prox_scad = providers.Factory(ProxSCAD, fix_param1=4, fix_param2=3.7)
    prox_mcp = providers.Factory(ProxMCP, fix_params=3.7)
    prox_tl1 = providers.Factory(ProxTransformedl1, fix_param1=1)
    prox_cl1 = providers.Factory(ProxCappedL1, fix_param=1)
    prox_cl1over2 = providers.Factory(ProxCapped1over2, fixparam=1)

    # L1psi proximal operators factory
    prox_1_1over2 = providers.Factory(
        ProxL1_1over2,
        n=n,
        gLen=gLen
    )

    prox_1_2over3 = providers.Factory(
        ProxL1_2over3,
        n=n,
        gLen=gLen
    )

    prox_1_scad = providers.Factory(
        GeneralProxL1Psi,
        n=n,
        gLen=gLen,
        subvec_prox=providers.Factory(
            L1_SCAD_SubvecProx,
            gLen=gLen,
            lamb=0,
            fix_param1=1,
            fix_param2=3.7,
            precompute=precompute
        ),
        num_samples=data_size
    )

    prox_1_mcp = providers.Factory(
        GeneralProxL1Psi,
        n=n,
        gLen=gLen,
        subvec_prox=providers.Factory(
            L1_MCP_SubvecProx,
            gLen=gLen,
            lamb=0,
            fix_param=3.7,
            precompute=precompute
        ),
        num_samples=data_size
    )

    prox_1_tl1 = providers.Factory(
        GeneralProxL1Psi,
        n=n,
        gLen=gLen,
        subvec_prox=providers.Factory(
            L1_Transformed,
            gLen=gLen,
            lamb=0,
            fix_param=4,
            precompute=precompute
        ),
        num_samples=data_size
    )

    # L2psi proximal operators factory
    prox_2_0 = providers.Factory(
        ProxL2_0,
        n=n, gLen=gLen
    )

    prox_2_1 = providers.Factory(
        ProxL2_1,
        n=n, gLen=gLen
    )

    prox_2_2over3 = providers.Factory(
        ProxL2_2over3,
        n=n, gLen=gLen
    )

    prox_2_1over2 = providers.Factory(
        ProxL2_1over2,
        n=n, gLen=gLen
    )

    prox_2_scad = providers.Factory(
        GeneralProxL2Psi, n=n, gLen=gLen, subvec_prox=
        prox_scad
    )

    prox_2_mcp = providers.Factory(
        GeneralProxL2Psi, n=n, gLen=gLen,
        subvec_prox=prox_mcp
    )

    prox_2_log = providers.Factory(
        ProxL2_LogSum,
        n=n, gLen=gLen, epsilon=1
    )

    prox_2_arctan = providers.Factory(
        ProxL2_Arctan,
        n=n, gLen=gLen, c=2
    )

    prox_2_tl1 = providers.Factory(
        GeneralProxL2Psi, n=n, gLen=gLen, subvec_prox=prox_tl1
    )

    prox_2_CL1 = providers.Factory(
        GeneralProxL2Psi, n=n, gLen=gLen, subvec_prox=prox_cl1
    )

    prox_2_CL1over2 = providers.Factory(
        GeneralProxL2Psi, n=n, gLen=gLen, subvec_prox=prox_cl1over2
    )
