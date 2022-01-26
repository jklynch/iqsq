from pathlib import Path

from iqsq import read_atom_positions


def test_read_atom_positions(tmp_path):
    atom_positions_path = tmp_path / Path("atom_positions.txt")
    with open(atom_positions_path, "wt") as f:
        f.write(
            """\
Na 0 0 1
Cl 0 1 0
"""
        )
    atom_positions_df = read_atom_positions(atom_positions_path=atom_positions_path)
    assert atom_positions_df.shape == (2, 3)
    assert atom_positions_df.index[0] == "Na"
    assert atom_positions_df.index[1] == "Cl"
    assert atom_positions_df.columns[0] == "x"
    assert atom_positions_df.columns[1] == "y"
    assert atom_positions_df.columns[2] == "z"
