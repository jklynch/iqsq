from io import StringIO

import numpy as np
import pandas as pd

import pytest

import iqsq


@pytest.fixture()
def scattering_factors_df():
    scattering_factors_df = pd.DataFrame(
        data=np.linspace(0, 1, num=3 * 9).reshape(3, 9),
        index=["H", "H1-", "He"],
    )

    return scattering_factors_df


@pytest.fixture()
def atom_distance_matrix_df():
    test_xyz = StringIO(
        """\
        H   0 0 1
        H1- 0 1 0
        He  1 0 0
        """
    )
    test_atom_positions_df = iqsq.read_atom_positions(atom_positions_path=test_xyz)
    test_atom_distance_matrix_df = iqsq.build_atom_distance_matrix(
        atom_positions_df=test_atom_positions_df
    )

    return test_atom_distance_matrix_df
