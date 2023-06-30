import cupy as cp
import numpy as np
import pandas as pd


def get_intermediate_results(
    scattering_factors_df,
    atom_distance_matrix_df,
    verbose=False
):
    """what a terrible name"""

    atom_distance_matrix = atom_distance_matrix_df.to_numpy()

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

    # work with only the scattering_factors for elements
    #   in the atom distance matrix
    # do a little extra work to keep the rows of
    #   reduced_scattering_factor_df in the same order as
    #   in scattering_factors_df
    elements_of_interest = [
        element
        for element in scattering_factors_df.index
        if element in unique_elements
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
    if verbose:
        approximate_storage_size = len(q_range) * np.product(atom_distance_matrix.shape)
        print(f"Q*A*A: {approximate_storage_size}")

    # Fi_df is a (atoms x q) array for each
    #   atom in atom_distance_matrixf_df
    Fi_df = q_element_constants_df.loc[atom_distance_matrix_df.index]
    if verbose:
        print(f"Fi_df.shape: {Fi_df.shape}")
        # print(f"Fi_df.data: {Fi_df}")
        print(f"Fi_df size: {np.product(Fi_df.shape)}")

    return (
        reduced_scattering_factors,
        atom_distance_matrix,
        qs,
        Fi_df.to_numpy()
    )


def calculate_iq(
    scattering_factors_df,
    atom_distance_matrix_df,
    qmin=0.6,
    qmax=20,
    qstep=0.05,
    gpu_dtype=np.float64,
    verbose=False,
):
    return calculate_iq_qrange(
        scattering_factors_df=scattering_factors_df,
        atom_distance_matrix_df=atom_distance_matrix_df,
        q_range=np.arange(qmin, qmax, qstep),
        gpu_dtype=gpu_dtype,
        verbose=verbose,
    )


def calculate_iq_qrange(
    scattering_factors_df,
    atom_distance_matrix_df,
    q_range,
    gpu_dtype=np.float64,
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
    #q_range = np.arange(qmin, qmax, qstep)
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
    if verbose:
        approximate_storage_size = len(q_range) * np.product(atom_distance_matrix.shape)
        print(f"Q*A*A: {approximate_storage_size}")

    # Fi_df is a (atoms x q) array for each
    #   atom in atom_distance_matrixf_df
    Fi_df = q_element_constants_df.loc[atom_distance_matrix_df.index]
    if verbose:
        print(f"Fi_df.shape: {Fi_df.shape}")
        # print(f"Fi_df.data: {Fi_df}")
        print(f"Fi_df size: {np.product(Fi_df.shape)}")

    # create a special atom distance matrix
    #   with a very small number on the diagonal
    #   rather than 0.0
    # the idea is that diagonal elements of sin(qA)/qA
    #   will be acceptably close to 1.0, which was
    #   previously assigned explicitly since sin(qA)/qA
    #   had NaN on the diagonal due to division by 0.0
    special_atom_distance_matrix = atom_distance_matrix
    special_atom_distance_matrix[np.diag_indices_from(atom_distance_matrix)] = 1.0e-10

    # move matrices to the GPU
    _q_Fi = cp.asarray(Fi_df.to_numpy(), dtype=gpu_dtype)

    _special_atom_distance_matrix = cp.asarray(
        special_atom_distance_matrix, dtype=gpu_dtype
    )
    _identity_matrix = cp.asarray(
        np.identity(n=atom_distance_matrix.shape[0], dtype=gpu_dtype)
    )

    _q_range = cp.asarray(q_range, dtype=gpu_dtype)
    Iq_sum_list = []
    for qi, (q, _q) in enumerate(zip(q_range, _q_range)):
        _Fi = cp.expand_dims(_q_Fi[:, qi], axis=1)
        # print(f"Fi.shape: {Fi.shape}")

        # print(np.shape(Fi))
        _FiFj = _Fi.T * _Fi
        # print(f"FiFj.shape: {FiFj.shape}")

        if q > 0.0:
            _q_atom_distance_matrix = _q * _special_atom_distance_matrix
            _sin_term_matrix = cp.sin(_q_atom_distance_matrix) / _q_atom_distance_matrix

            # at this point the diagonal elements of sin_term_matrix
            # are all approximately 1.0
        elif q == 0.0:
            _sin_term_matrix = _identity_matrix
        else:
            raise ValueError(f"q is less than zero: {_q}")

        _Iq = _FiFj * _sin_term_matrix

        # sum Iq for each pair only once
        # Iq_sum_list.append(np.sum(Iq[np.triu_indices(Iq.shape[0])]))

        # sum Iq for each pair twice, except for "self" pairs such as (O_0, O_0)
        # (pairs from the diagonal of the distance matrix)
        Iq_sum_list.append(cp.sum(_Iq))

    # print("Iq.shape")
    # print(Iq.shape)
    qIq = np.column_stack((q_range, [_Iq_sum.get() for _Iq_sum in Iq_sum_list]))
    # print("qIq")
    # print(qIq)

    return qIq
