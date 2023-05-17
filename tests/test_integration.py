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

    measurements = {}
    for col in example_input_data.columns:
        x = lab_toolbox.Measurement(col, example_input_data[col])
        measurements[x.name] = x


    for id, data in matlab_integral_data.iterrows():
        start, end, imax, integral, int_imax = data
        example_baseline_filtered_data['Time/s'] = example_input_data['Time/s']
        example_baseline_filtered_data.set_index('Time/s', inplace=True)


        x = np.array(example_baseline_filtered_data.index)
        y = np.array(example_baseline_filtered_data[id])

        x_subset = x[
            np.where((x >= start) & (x <= end))
        ]
        y_subset = y[
            np.where((x >= start) & (x <= end))
        ]
        print(y_subset)
        assert len(x_subset) == len(y_subset)

        calc_integral = lab_toolbox.calculate_integral(y_subset, x_subset,
                                                       'trapz')
        assert calc_integral == integral