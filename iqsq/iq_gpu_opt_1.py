import cupy as cp
import numpy as np
import pandas as pd


def calculate_iq(
    scattering_factors_df,
    atom_distance_matrix_df,
    qmin=0.6,
    qmax=20,
    qstep=0.05,
    verbose=False,
):
    # atom_element looks like
    #   ['O', 'Co', 'O', 'O', 'O', 'O', 'O', 'Co', 'Co',...]
    atom_element = atom_distance_matrix_df.index
    if verbose:
        print("atom_element")
        print(atom_element)

    # set(atom_element) looks like {'O', 'Co'}
    unique_elements = set(atom_element)
    if verbose:
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
    if verbose:
        print(f"reduced_scattering_factors.shape: {reduced_scattering_factors.shape}")

    # loop on q
    q_range = np.arange(qmin, qmax, qstep)
    if verbose:
        # print(f"q_range: {q_range}")
        print(f"q_range.shape: {q_range.shape}")

    # how much memory are we looking at for all the q?
    q_reduced_scattering_size = len(q_range) * np.product(
        reduced_scattering_factors.shape
    )
    if verbose:
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
    if verbose:
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
    if verbose:
        print(f"Q*A*A: {approximate_storage_size}")

    # Fi_df is a (atoms x q) array for each
    #   atom in atom_distance_matrixf_df
    Fi_df = q_element_constants_df.loc[atom_distance_matrix_df.index]
    if verbose:
        print(f"Fi_df.shape: {Fi_df.shape}")
        # print(f"Fi_df.data: {Fi_df}")
        print(f"Fi_df size: {np.product(Fi_df.shape)}")

    q_Fi = cp.asarray(Fi_df.to_numpy())

    atom_distance_matrix = cp.asarray(atom_distance_matrix)
    atom_distance_matrix_diagonal_indices = cp.diag_indices_from(atom_distance_matrix)

    Iq_sum_list = []
    for qi, q in enumerate(q_range):
        Fi = cp.expand_dims(q_Fi[:, qi], axis=1)
        # print(f"Fi.shape: {Fi.shape}")

        # print(np.shape(Fi))
        FiFj = Fi.T * Fi
        # print(f"FiFj.shape: {FiFj.shape}")

        if q > 0.0:
            # the next line will cause a warning like this:
            #   ../site-packages/ipykernel_launcher.py:???:
            #   RuntimeWarning: invalid value encountered in true_divide
            # but this is not an error, it tells us the sin_term_matrix has
            # NaN on the diagonal which will be corrected on the following line

            q_atom_distance_matrix = q * atom_distance_matrix
            sin_term_matrix = cp.sin(q_atom_distance_matrix) / q_atom_distance_matrix

            # set the diagonal elements to 1.0
            #   very slow for large matrices
            #   sin_term_matrix[cp.isnan(sin_term_matrix)] = 1.0
            sin_term_matrix[atom_distance_matrix_diagonal_indices] = 1.0
            # print("sin_term_matrix")
            # print(sin_term_matrix)
        elif q == 0.0:
            sin_term_matrix = cp.eye_like(atom_distance_matrix)
        else:
            raise ValueError(f"q is less than zero: {q}")

        Iq = FiFj * sin_term_matrix

        # sum Iq for each pair only once
        # Iq_sum_list.append(np.sum(Iq[np.triu_indices(Iq.shape[0])]))

        # sum Iq for each pair twice, except for "self" pairs such as (O_0, O_0)
        # (pairs from the diagonal of the distance matrix)
        # cp.sum(Iq) returns a 1-element array
        # unpack that now
        Iq_sum_list.append(cp.sum(Iq))

    qIq = np.column_stack((q_range, [Iq_sum.get() for Iq_sum in Iq_sum_list]))

    return qIq, Fi.get()
