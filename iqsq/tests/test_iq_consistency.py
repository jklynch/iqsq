from iqsq import iq_cpu_naive


def test_iq_naive(scattering_factors, atom_distance_matrix):
    iq_cpu_naive.calculate_iq(
        scattering_factors_df=scattering_factors,
        atom_distance_matrix_df=atom_distance_matrix,
        qmin=0,
        qmax=0.5,
        qstep=0.1,
    )
