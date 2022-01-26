import pandas as pd

from scipy.spatial.distance import cdist


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def read_aff_elements(path, *args, **kwargs):
    return pd.read_csv(filepath_or_buffer=path, *args, **kwargs)


def read_aff_parameters(path, *args, **kwargs):
    return pd.read_csv(filepath_or_buffer=path, *args, **kwargs)


def read_atom_positions(atom_positions_path):
    """Load data from .xyz file.
    no header
    """
    atom_positions_df = pd.read_table(
        filepath_or_buffer=atom_positions_path,
        header=None,
        names=["x", "y", "z"],
        index_col=0,
        delim_whitespace=True,
    )
    # atom_positions_df.columns = ["x", "y", "z"]
    return atom_positions_df


def build_atom_distance_matrix(atom_positions_df):
    """Construct a distance matrix from atom positions.

    Parameters
    ----------
    atom_positions_df: pandas.DataFrame
        a 3xN dataframe with index x,y,z and one column per atom
        for example:
             x  y  z
        Na   0  0  1
        Cl   0  1  0
        Na   1  0  0

    Returns
    -------
    pandas.DataFrame NxN distance matrix with atom names on index and columns
        for example:
            Na     Cl     Na
        Na  0.0    1.414  1.414
        Cl  1.414  0.0    1.414
        Na  1.414  1.414  0.0
    """
    atom_distance_matrix = cdist(
        atom_positions_df.to_numpy(), atom_positions_df.to_numpy()
    )
    atom_distance_matrix_df = pd.DataFrame(
        data=atom_distance_matrix,
        columns=atom_positions_df.index,
        index=atom_positions_df.index,
    )
    # set index name to None, otherwise it is "0" and that looks
    #   odd when the dataframe is printed
    atom_distance_matrix_df.index.name = None
    return atom_distance_matrix_df
