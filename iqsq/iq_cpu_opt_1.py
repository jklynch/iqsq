import numpy as np

import iqsq.q_constants


def calculate_iq(scattering_factors_df, atom_distance_matrix_df, qmin, qmax, qstep):
    """
    Parameters
    ----------
    scattering_factors_df: pandas.DataFrame
      9 rows
      atoms on columns
    """
    # atom_element looks like
    #   ['O', 'Co', 'O', 'O', 'O', 'O', 'O', 'Co', 'Co',...]
    # atom_element = np.array([
    #    element_number.split('_')[0]
    #    for element_number
    #    in atom_distance_matrix_df.columns
    # ])
    atom_element = atom_distance_matrix_df.index
    print("atom_element")
    print(atom_element)

    # set(atom_element) looks like {'O', 'Co'}
    unique_elements = set(atom_element)
    print(f"unique elements: {unique_elements}")

    atom_distance_matrix = atom_distance_matrix_df.to_numpy()
    Iq_sum_list = []

    # loop on q
    q_range = np.arange(qmin, qmax, qstep)
    print(f"q_range: {q_range}")
    print(f"q_range.shape: {q_range.shape}")
    if True:
        # for q in q_range:
        qs = np.expand_dims(q_range, axis=(1, 2))

        q_element_constants_df = iqsq.q_constants.optimize_1(
            qs=q_range,
            atom_distance_matrix_df=atom_distance_matrix_df,
            scattering_factors_df=scattering_factors_df,
        )

        # Fi_df is a (q x atoms) array for each
        #   atom in atom_distance_matrixf_df
        Fi_df = q_element_constants_df.loc[atom_distance_matrix_df.index]

        Fi_matrix = Fi_df.to_numpy()
        print("Fi_matrix")
        print(Fi_matrix)

        Fi = np.expand_dims(Fi_matrix.T, axis=2)
        print("Fi.shape")
        print(Fi.shape)

        Fj = np.expand_dims(Fi_matrix.T, axis=1)
        print("Fj.shape")
        print(Fj.shape)

        FiFj = np.matmul(Fi, Fj)
        print("FiFj.shape")
        print(FiFj.shape)

        # the next line will cause a warning like this:
        #   ../site-packages/ipykernel_launcher.py:???:
        #   RuntimeWarning: invalid value encountered in true_divide
        # but this is not an error, it tells us the sin_term_matrix has
        # NaN on the diagonal which will be corrected on the following line
        # this line is the bottleneck when the number of atoms gets large
        sin_term_matrix = np.sin(qs * atom_distance_matrix) / (
            qs * atom_distance_matrix
        )
        # set the diagonal elements to 1.0
        # need to hit all diagonals!
        sin_term_matrix[np.diag_indices(sin_term_matrix.shape[0])] = 1.0
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
    print(f"Iq_sum_list: {Iq_sum_list}")
    qIq = np.column_stack((q_range, Iq_sum_list))
    # print("qIq")
    # print(qIq)

    return qIq, Fi
