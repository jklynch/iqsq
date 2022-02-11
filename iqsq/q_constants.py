import numpy as np
import pandas as pd


def naive(scattering_factors_df, qs):
    """
    qs has shape Nx1
    scattering_coefficients_df is atoms x 9
    """
    element_count = scattering_factors_df.shape[0]
    q_count = qs.shape[0]
    q_constants = np.zeros((q_count, element_count))
    for q_i, q in enumerate(qs):
        scattering_coefficients = scattering_factors_df.to_numpy()
        fi1 = scattering_coefficients[:, 0] * np.exp(
            -scattering_coefficients[:, 1] * ((q / (4 * np.pi)) ** 2)
        )
        fi2 = scattering_coefficients[:, 2] * np.exp(
            -scattering_coefficients[:, 3] * ((q / (4 * np.pi)) ** 2)
        )
        fi3 = scattering_coefficients[:, 4] * np.exp(
            -scattering_coefficients[:, 5] * ((q / (4 * np.pi)) ** 2)
        )
        fi4 = scattering_coefficients[:, 6] * np.exp(
            -scattering_coefficients[:, 7] * ((q / (4 * np.pi)) ** 2)
        )
        fic = scattering_coefficients[:, 8]

        # print("atom_element == ", element)
        # print(atom_element == element)
        q_constants[q_i, :] = fi1 + fi2 + fi3 + fi4 + fic

    q_constants_df = pd.DataFrame(
        data=q_constants.T, index=scattering_factors_df.index, columns=qs
    )

    return q_constants_df


def optimize_1(qs, atom_distance_matrix_df, scattering_factors_df):
    # qs must be a 1-dimensional array

    # create another array of qs with extra dimensions
    #   to be commensurate with the scattering factors array
    # qs_expanded = np.expand_dims(qs, axis=(1, 2))
    # print(f"qs_expanded.shape: {qs_expanded.shape}")

    # work with only the scattering_factors for the elements
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

    # print(f"reduced_scattering_factors.shape: {reduced_scattering_factors.shape}")
    # print(f"reduced_scattering_factors[:, 1:9:2].shape: {reduced_scattering_factors[:, 1:9:2].shape}")
    # a = reduced_scattering_factors[:, 1:9:2] * qs_expanded
    # print(f"(reduced_scattering_factors[:, 1:9:2] * qs_expanded).shape: {a.shape}")

    # reduced_scattering_factors has shape (elements, 9)
    # we need to expand the shape of qs from (Q, ) to (Q, 1, 1)
    #  so that reduced_scattering_factors[:, 1:9:2] * qs_expanded
    #  has shape (E, 4) * (Q, 1, 1) -> (Q, E, 4)
    #  where E is the number of elements and Q is the number of qs
    q_element_constants = (
        np.sum(
            reduced_scattering_factors[:, 0:8:2]
            * np.exp(
                -1
                * reduced_scattering_factors[:, 1:9:2]
                * ((np.expand_dims(qs, axis=(1, 2)) / (4 * np.pi)) ** 2)
            ),
            axis=2,
        )
        + reduced_scattering_factors[:, 8]
    )

    q_element_constants_df = pd.DataFrame(
        data=q_element_constants.T,
        index=reduced_scattering_factors_df.index,
        columns=qs,
    )

    return q_element_constants_df
