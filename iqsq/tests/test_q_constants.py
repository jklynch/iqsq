import numpy as np

import iqsq.q_constants


def test_consistency(scattering_factors_df, atom_distance_matrix_df):
    qs = np.linspace(0, 0.5, num=5)

    naive_q_constants = iqsq.q_constants.naive(
        scattering_factors_df=scattering_factors_df, qs=qs
    )

    opt_1_q_constants = iqsq.q_constants.optimize_1(
        scattering_factors_df=scattering_factors_df,
        atom_distance_matrix_df=atom_distance_matrix_df,
        qs=qs,
    )

    assert np.all(np.equal(naive_q_constants, opt_1_q_constants))
