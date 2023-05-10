from pydantic import BaseSettings


class Constants(BaseSettings):
    DEFAULT_POINT_SIZE: int = 0
    DEFAULT_POINT_COLOUR: str = '#a3a7e4'


constants = Constants()