from pydantic import BaseSettings


class Constants(BaseSettings):
    DEFAULT_POINT_SIZE: int = 0
    DEFAULT_POINT_COLOUR: str = '#a3a7e4'
    INTEGRAL_PEAK_COLOUR: str = '#FF6692'
    INTEGRAL_PEAK_SIZE: int = 20

    SELECTED_POINT_COLOUR: str = '#2F4F4F'
    SELECTED_POINT_SIZE: int = 20
    FILENAME_RAW_DATA: str = 'RawDatOut_301_b.csv'
    FILENAME_BASELINE_SUBTRACTED_DATA: str = 'bs_301_b.csv'
    FILENAME_SUMMARY_DATA: str = 'Output_301_b.csv'

    FILE_HEADER: str = 'Time/s, i1/A, i2/A, i3/A, i4/A, i5/A, i6/A, i7/A, i8/A'

constants = Constants()
