import pandas as pd
import sys
import os
import numpy as np


# Add the parent folder to the Python module search path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
)
import lab_toolbox


def test_integration_by_x_points(
    example_input_data: pd.DataFrame,
    example_baseline_filtered_data: pd.DataFrame,
    matlab_integral_data: pd.DataFrame
):
    ''' Test integration is equivalent to original MATLAB code

    Using the outputs of the SOIL lab's workflow, test the integration
    method is equivalent.
    '''

    for col in example_input_data.columns:
        x = lab_toolbox.Measurement(col, example_input_data[col])

    example_baseline_filtered_data.set_index('Time/s', inplace=True)
    for id, data in matlab_integral_data.iterrows():
        start, end, imax, integral, int_imax = data

        x = np.array(example_baseline_filtered_data.index)
        y = np.array(example_baseline_filtered_data[id])

        x_subset = x[
            np.where((x >= start) & (x <= end))
        ]
        y_subset = y[
            np.where((x >= start) & (x <= end))
        ]

        assert len(x_subset) == len(y_subset)

        calc_integral = lab_toolbox.integrate_coulomb_as_mole(
            y_subset, x_subset, 'trapz'
        )

        # Assert the value of calc_integral is equivalent to the MATLAB.
        # Minor differences lower than the input significant figure expected.
        assert np.isclose(calc_integral, integral), (
            f"Integration is not close in measurement: '{id}'"
        )
