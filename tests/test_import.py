import pandas as pd
import sys
import os

# Add the parent folder to the Python module search path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
)
from lab_toolbox import lab_toolbox

RESOURCES_DIR = os.path.join(os.getcwd(), "tests", "resources")


def test_find_header_start() -> None:
    """Test find_header_start() returns the correct value"""

    # Read the example input data
    header_row = lab_toolbox.find_header_start(
        os.path.join(RESOURCES_DIR, "Example_input_8col.txt")
    )

    assert header_row == 25, "Header row is not 25"

    # Read the example input data
    header_row = lab_toolbox.find_header_start(
        os.path.join(RESOURCES_DIR, "Example_input_6col.txt")
    )

    assert header_row == 62


def test_import_dynamic_column_lengths(
    example_input_data_6col: pd.DataFrame,
    example_input_data_8col: pd.DataFrame,
):
    """Test import of data with 6 or 8 column lengths works as expected"""

    for col in example_input_data_8col.columns:
        x = lab_toolbox.Measurement(col, example_input_data_8col[col])
        assert x is not None
        assert len(x.raw_data) > 0, "No data in column"

    for col in example_input_data_6col.columns:
        x = lab_toolbox.Measurement(col, example_input_data_6col[col])
        assert x is not None
        assert len(x.raw_data) > 0, "No data in column"
        assert x.raw_data.name not in ["i4/A", "i8/A"], "Unexpected column"
