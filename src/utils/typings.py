from typing import Tuple

from beartype import beartype
from beartype.vale import Is
from typing_extensions import Annotated


@beartype
def int_greater_or_equal(value: int):
    return Annotated[int, Is[lambda v: v >= value]]


int_non_negative = int_greater_or_equal(0)


@beartype
def int_in_range(min_value: int, max_value: int):
    return Annotated[int, Is[lambda v: min_value <= v <= max_value]]


@beartype
def float_greater_or_equal(value: float):
    return Annotated[float, Is[lambda v: v >= value]]


float_non_negative = float_greater_or_equal(0.0)


@beartype
def float_in_range(min_value: float, max_value: float):
    return Annotated[float, Is[lambda v: min_value <= v <= max_value]]


class ImageResolution:
    height: int
    width: int

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def to_hw_tuple(self) -> Tuple[int, int]:
        return self.height, self.width

    def to_wh_tuple(self) -> Tuple[int, int]:
        return self.width, self.height

    def __eq__(self, other):
        return self.height == other.height and self.width == other.width
