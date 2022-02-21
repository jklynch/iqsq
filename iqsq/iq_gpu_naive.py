import cupy as cp
import numpy as np


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

    atom_distance_matrix = cp.asarray(atom_distance_matrix_df.to_numpy())
    Iq_sum_list = []

    # we can allocate this vector once and reuse it
    # Fi needs shape (1, atoms count) so we can calculate an outer product
    #   Fi.T * Fi
    Fi = cp.zeros((1, len(atom_element)), dtype=np.float64)
    print("initial Fi")
    print(Fi)

    # loop on q
    q_range = np.arange(qmin, qmax, qstep)
    print(f"q_range: {q_range}")
    for q in q_range:
        # can we calculate some of this ahead?
        for element in unique_elements:
            scattering_values = scattering_factors_df.loc[element, :]
            fi1 = scattering_values[0] * np.exp(
                -scattering_values[1] * ((q / (4 * np.pi)) ** 2)
            )
            fi2 = scattering_values[2] * np.exp(
                -scattering_values[3] * ((q / (4 * np.pi)) ** 2)
            )
            fi3 = scattering_values[4] * np.exp(
                -scattering_values[5] * ((q / (4 * np.pi)) ** 2)
            )
            fi4 = scattering_values[6] * np.exp(
                -scattering_values[7] * ((q / (4 * np.pi)) ** 2)
            )
            fic = scattering_values[8]

            # print(atom_element == element)
            Fi[0, atom_element == element] = fi1 + fi2 + fi3 + fi4 + fic

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
            # this line is the bottleneck when the number of atoms gets large
            sin_term_matrix = np.sin(q * atom_distance_matrix) / (
                q * atom_distance_matrix
            )
            # set the diagonal elements to 1.0
            sin_term_matrix[np.diag_indices(sin_term_matrix.shape[0])] = 1.0
            # print("sin_term_matrix")
            # print(sin_term_matrix)
        elif q == 0.0:
            sin_term_matrix = cp.eye_like(atom_distance_matrix)
        else:
            # q is less than 0.0
            raise ValueError(f"q is less than zero: {q}")

        Iq = FiFj * sin_term_matrix

        # sum Iq for each pair only once
        # Iq_sum_list.append(np.sum(Iq[np.triu_indices(Iq.shape[0])]))

        # sum Iq for each pair twice, except for "self" pairs such as (O_0, O_0)
        # (pairs from the diagonal of the distance matrix)
        Iq_sum_list.append(np.sum(Iq))

    # print("Iq.shape")
    # print(Iq.shape)
    qIq = np.column_stack((q_range, [s.get() for s in Iq_sum_list]))
    # print("qIq")
    # print(qIq)

    return qIq, Fi
