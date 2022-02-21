import numpy as np
import pandas as pd

from toolz.itertoolz import partition_all


def calculate_iq(
    scattering_factors_df,
    atom_distance_matrix_df,
    qmin,
    qmax,
    qstep,
    q_partition_size=500,
):
    """
    Parameters
    ----------
    scattering_factors_df: pandas.DataFrame
      9 rows
      atoms on columns
    """
    # atom_element looks like
    #   ['O', 'Co', 'O', 'O', 'O', 'O', 'O', 'Co', 'Co',...]
    atom_element = atom_distance_matrix_df.index
    print("atom_element")
    print(atom_element)

    # set(atom_element) looks like {'O', 'Co'}
    unique_elements = set(atom_element)
    print(f"unique elements: {unique_elements}")

    atom_distance_matrix = atom_distance_matrix_df.to_numpy()

    # work with only the scattering_factors for elements
    #   in the atom distance matrix
    # do a little extra work to keep the rows of
    #   reduced_scattering_factor_df in the same order as
    #   in scattering_factors_df
    elements_of_interest = [
        element
        for element in scattering_factors_df.index
        if element in atom_distance_matrix_df.index
    ]
    reduced_scattering_factors_df = scattering_factors_df.loc[elements_of_interest]
    reduced_scattering_factors = reduced_scattering_factors_df.to_numpy()
    print(f"reduced_scattering_factors.shape: {reduced_scattering_factors.shape}")

    # loop on q
    q_range = np.arange(qmin, qmax, qstep)
    # print(f"q_range: {q_range}")
    print(f"q_range.shape: {q_range.shape}")

    # how much memory are we looking at for all the q?
    q_reduced_scattering_size = len(q_range) * np.product(
        reduced_scattering_factors.shape
    )
    print(f"q_reduced_scattering_size: {q_reduced_scattering_size}")

    # we need to expand the shape of q_range from (Q, ) to (Q, 1, 1)
    #  so that reduced_scattering_factors[:, 1:9:2] * qs_expanded
    #  has shape (E, 4) * (Q, 1, 1) -> (Q, E, 4)
    #  where E is the number of elements and Q is the number of qs
    qs = np.expand_dims(q_range, axis=(1, 2))

    # reduced_scattering_factors has shape (elements, 9)
    q_element_constants = (
        np.sum(
            reduced_scattering_factors[:, 0:8:2]
            * np.exp(
                -1 * reduced_scattering_factors[:, 1:9:2] * ((qs / (4 * np.pi)) ** 2)
            ),
            axis=2,
        )
        + reduced_scattering_factors[:, 8]
    )
    print(f"q_element_constants.shape: {q_element_constants.shape}")
    print(f"q_element_constants size: {np.prod(q_element_constants.shape)}")

    q_element_constants_df = pd.DataFrame(
        data=q_element_constants.T,
        index=reduced_scattering_factors_df.index,
        columns=q_range,
    )

    # the approximate storage needed will be
    #   Q * A * A
    approximate_storage_size = len(q_range) * np.product(atom_distance_matrix.shape)
    print(f"Q*A*A: {approximate_storage_size}")

    # Fi_df is a (atoms x q) array for each
    #   atom in atom_distance_matrixf_df
    Fi_df = q_element_constants_df.loc[atom_distance_matrix_df.index]
    print(f"Fi_df.shape: {Fi_df.shape}")
    print(f"Fi_df size: {np.product(Fi_df.shape)}")

    # try working with subsets of Fi_df
    Iq_sums = list()
    for q_index_subset in partition_all(q_partition_size, range(Fi_df.shape[1])):
        q_index_min = q_index_subset[0]
        q_index_max = q_index_subset[-1]
        # Fi_matrix = Fi_df.to_numpy()
        Fi_matrix = Fi_df.iloc[:, q_index_min : q_index_max + 1].to_numpy()
        print(f"Fi_matrix.shape: {Fi_matrix.shape}")

        Fi = np.expand_dims(Fi_matrix.T, axis=2)
        print(f"Fi.shape: {Fi.shape}")

        Fj = np.expand_dims(Fi_matrix.T, axis=1)
        print(f"Fj.shape: {Fj.shape}")

        FiFj = np.matmul(Fi, Fj)
        print(f"FiFj.shape: {FiFj.shape}")
        print(f"FiFj size: {np.product(FiFj.shape)}")

        # the next line will cause a warning like this:
        #   ../site-packages/ipykernel_launcher.py:???:
        #   RuntimeWarning: invalid value encountered in true_divide
        # but this is not an error, it tells us the sin_term_matrix has
        # NaN on the diagonal which will be corrected on the following line
        # this line is the bottleneck when the number of atoms gets large
        qs_atom_distance_matrix = (
            qs[q_index_min : q_index_max + 1, :] * atom_distance_matrix
        )
        sin_term_matrix = np.sin(qs_atom_distance_matrix) / qs_atom_distance_matrix
        # print(f"sin_term_matrix:\n{sin_term_matrix}")
        # set the diagonal elements to 1.0
        # need to hit all diagonals!
        # sin_term_matrix[np.diag_indices(sin_term_matrix.shape[0])] = 1.0

        sin_term_matrix[np.isnan(sin_term_matrix)] = 1.0

        # print("sin_term_matrix")
        # print(sin_term_matrix)

        # elif q == 0.0:
        #    sin_term_matrix = np.eye_like(atom_distance_matrix)
        # else:
        #    # q is less than 0.0
        #    raise ValueError(f"q is less than zero: {q}")

        Iq = FiFj * sin_term_matrix

        # sum Iq for each pair only once
        # Iq_sum_list.append(np.sum(Iq[np.triu_indices(Iq.shape[0])]))

        # sum Iq for each pair twice, except for "self" pairs such as (O_0, O_0)
        # (pairs from the diagonal of the distance matrix)
        # Iq_sum_list.append(np.sum(Iq))

        print("Iq.shape")
        print(Iq.shape)

        Iq_sum = np.sum(Iq, axis=(1, 2))
        print(f"Iq_sum.shape:\n{Iq_sum.shape}")
        Iq_sums.append(Iq_sum)

    # qIq = np.column_stack((q_range, Iq_sums))
    qIq = 0
    # print("qIq")
    # print(qIq)

    return qIq
