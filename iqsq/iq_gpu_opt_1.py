import cupy as cp
import numpy as np


def calculate_iq(
    scattering_factors_df, atom_distance_matrix_df, qmin=0.6, qmax=20, qstep=0.05
):
    # atom_element looks like
    #   ['O', 'Co', 'O', 'O', 'O', 'O', 'O', 'Co', 'Co',...]
    # atom_element = np.array([
    #    element_number.split('_')[0]
    #    for element_number
    #    in atom_distance_matrix_df.columns
    # ])
    atom_element = atom_distance_matrix_df.columns
    print("atom_element")
    print(atom_element)

    # set(atom_element) looks like {'O', 'Co'}
    unique_elements = set(atom_element)
    print(f"unique elements: {unique_elements}")

    atom_distance_matrix = cp.asarray(atom_distance_matrix_df.to_numpy())
    Iq_sum_list = []

    # we can allocate this vector once and reuse it
    # Fi needs shape (1, atoms count) so we can calculate an outer product
    #   Fi.T * Fi
    Fi = cp.zeros((1, len(atom_element)), dtype=np.float64)
    print(f"Fi shape: {Fi.shape}")

    # calculate some vectors that never change
    Si1 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Gi1 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Si2 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Gi2 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Si3 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Gi3 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Si4 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Gi4 = cp.zeros((1, len(atom_element)), dtype=np.float64)
    Sic = cp.zeros((1, len(atom_element)), dtype=np.float64)
    for element in unique_elements:
        scattering_values = cp.asarray(scattering_factors_df[element])
        Si1[0, atom_element == element] = scattering_values[0]
        Gi1[0, atom_element == element] = cp.exp(
            -scattering_values[1] * (1.0 / (4.0 * cp.pi)) ** 2
        )
        Si2[0, atom_element == element] = scattering_values[2]
        Gi2[0, atom_element == element] = cp.exp(
            -scattering_values[3] * (1.0 / (4.0 * cp.pi)) ** 2
        )
        Si3[0, atom_element == element] = scattering_values[4]
        Gi3[0, atom_element == element] = cp.exp(
            -scattering_values[5] * (1.0 / (4.0 * cp.pi)) ** 2
        )
        Si4[0, atom_element == element] = scattering_values[6]
        Gi4[0, atom_element == element] = cp.exp(
            -scattering_values[7] * (1.0 / (4.0 * cp.pi)) ** 2
        )
        Sic[0, atom_element == element] = scattering_values[8]

    sin_matrix_diagonal_index = cp.diag_indices_from(atom_distance_matrix)

    # loop on q
    q_range = np.arange(qmin, qmax, qstep)
    print(f"q_range: {q_range}")
    for q in q_range:
        Fi = (
            Si1 * cp.power(Gi1, q ** 2)
            + Si2 * cp.power(Gi2, q ** 2)
            + Si3 * cp.power(Gi3, q ** 2)
            + Si4 * cp.power(Gi4, q ** 2)
            + Sic
        )
        # print(f"Fi:")
        # print(Fi)
        # print(np.shape(Fi))
        FiFj = Fi.T * Fi
        # print("FiFj")
        # print(FiFj)

        if q > 0.0:
            # the next line will cause a warning like this:
            #   ../site-packages/ipykernel_launcher.py:???:
            #   RuntimeWarning: invalid value encountered in true_divide
            # but this is not an error, it tells us the sin_term_matrix has
            # NaN on the diagonal which will be corrected on the following line

            # sin_term_matrix = np.sin(q*atom_distance_matrix) / (q*atom_distance_matrix)
            q_atom_distance_matrix = q * atom_distance_matrix
            sin_term_matrix = cp.sin(q_atom_distance_matrix) / q_atom_distance_matrix

            # set the diagonal elements to 1.0
            sin_term_matrix[sin_matrix_diagonal_index] = 1.0
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
        Iq_sum_list.append(cp.sum(Iq))

    # print("Iq.shape")
    # print(Iq.shape)
    qIq = np.column_stack((q_range, [s.get() for s in Iq_sum_list]))
    # print("qIq")
    # print(qIq)

    return qIq, Fi.get()
