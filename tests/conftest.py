import pandas as pd
import pytest
import os


RESOURCES_DIR = os.path.join(os.getcwd(), 'tests', 'resources')


@pytest.fixture(scope="session")
def matlab_integral_data():
    ''' Read integral data from MATLAB outputs '''

    df = pd.read_csv(
        os.path.join(RESOURCES_DIR, 'matlab_integral_output.csv')
    )
    df.set_index('measurement', inplace=True)

    return df


@pytest.fixture(scope="session")
def example_input_data():
    ''' Read example input data '''

    df = pd.read_csv(
        os.path.join(RESOURCES_DIR, 'Example_input.txt'),
        header=25
    )
    # df.set_index('Time/s', inplace=True)  # Set index to the time column
    df.columns = df.columns.str.strip()   # Strip whitespace from the columns

    return df


@pytest.fixture(scope="session")
def example_baseline_filtered_data():
    ''' Read example baseline data '''

    df = pd.read_csv(
        os.path.join(RESOURCES_DIR, 'matlab_baseline_filtered.csv')
    )

    return df
