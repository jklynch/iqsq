from pathlib import Path

import numpy as np

from iqsq import build_atom_distance_matrix, read_atom_positions


def test_build_atom_distance_matrix(tmp_path):
    atom_positions_path = tmp_path / Path("atom_positions.txt")
    with open(atom_positions_path, "wt") as f:
        f.write(
            """\
Na 0 0 1
Cl 0 1 0
"""
        )
    atom_positions_df = read_atom_positions(atom_positions_path=atom_positions_path)
    atom_distance_matrix_df = build_atom_distance_matrix(
        atom_positions_df=atom_positions_df
    )
    assert atom_distance_matrix_df.shape == (2, 2)
    assert atom_distance_matrix_df.index[0] == "Na"
    assert atom_distance_matrix_df.index[1] == "Cl"
    assert atom_distance_matrix_df.columns[0] == "Na"
    assert atom_distance_matrix_df.columns[1] == "Cl"
    assert atom_distance_matrix_df.iloc[0, 0] == 0.0
    assert atom_distance_matrix_df.iloc[1, 0] == np.sqrt(2)
    assert atom_distance_matrix_df.iloc[0, 1] == np.sqrt(2)
    assert atom_distance_matrix_df.iloc[1, 1] == 0.0
