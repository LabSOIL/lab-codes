from pydantic import BaseSettings


class Constants(BaseSettings):
    DEFAULT_POINT_SIZE: int = 0
    DEFAULT_POINT_COLOUR: str = '#a3a7e4'
    INTEGRAL_PEAK_COLOUR: str = '#FF6692'
    INTEGRAL_PEAK_SIZE: int = 20

    FILENAME_RAW_DATA: str = 'RawDatOut_301_b.csv'
    FILENAME_BASELINE_SUBTRACTED_DATA: str = 'bs_301_b.csv'


constants = Constants()
