from pathlib import Path

from iqsq import read_aff_elements


def test_read_aff_elements(tmp_path):
    test_aff_elements_path = tmp_path / Path("aff_elements.txt")
    with open(test_aff_elements_path, "wt") as f:
        f.write(
            """\
H
H1-
He
Li
Li1+
"""
        )
    aff_elements_df = read_aff_elements(path=test_aff_elements_path, header=None)

    assert aff_elements_df.shape == (5, 1)
    assert aff_elements_df.iloc[0, 0] == "H"
    assert aff_elements_df.iloc[-1, 0] == "Li1+"


def test_read_aff_parameters(tmp_path):
    test_aff_parameters_path = tmp_path / Path("aff_parameters.txt")
    with open(test_aff_parameters_path, "wt") as f:
        f.write(
            """\
0.489918	20.659300	0.262003	7.740390	0.196767	49.551900	0.049879	2.201590	0.001305
0.897661	53.136800	0.565616	15.187000	0.415815	186.576000	0.116973	3.567090	0.002389
0.873400	9.103700	0.630900	3.356800	0.311200	22.927600	0.178000	0.982100	0.006400
1.128200	3.954600	0.750800	1.052400	0.617500	85.390500	0.465300	168.261000	0.037700
0.696800	4.623700	0.788800	1.955700	0.341400	0.631600	0.156300	10.095300	0.016700
"""
        )
    aff_parameters_df = read_aff_elements(
        path=test_aff_parameters_path, header=None, delim_whitespace=True
    )

    assert aff_parameters_df.shape == (5, 9)
    assert aff_parameters_df.iloc[0, 0] == 0.489918
    assert aff_parameters_df.iloc[-1, -1] == 0.016700
