import numpy as np

from iqsq import iq_cpu_naive, iq_cpu_opt_1


def test_iq_naive(scattering_factors_df, atom_distance_matrix_df):
    iq_naive = iq_cpu_naive.calculate_iq(
        scattering_factors_df=scattering_factors_df,
        atom_distance_matrix_df=atom_distance_matrix_df,
        qmin=0,
        qmax=0.5,
        qstep=0.1,
    )

    iq_opt_1 = iq_cpu_opt_1.calculate_iq(
        scattering_factors_df=scattering_factors_df,
        atom_distance_matrix_df=atom_distance_matrix_df,
        qmin=0,
        qmax=0.5,
        qstep=0.1,
    )

    assert np.all(np.equal(iq_naive, iq_opt_1))
