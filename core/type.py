def positive_float(input: str) -> float:
    value = float(input)
    if value <= 0:
        raise ValueError(f"{value} is not a positive floating point number")

    return value


def positive_int(input: str) -> int:
    value = int(input)
    if value <= 0:
        raise ValueError(f"{value} is not a positive integer")

    return value


def non_negative_int(input: str) -> int:
    value = int(input)
    if value < 0:
        raise ValueError(f"{value} is negative integer")

    return value


def percent_number(input: str) -> float:
    value = float(input)
    if value < 0 or value > 1:
        raise ValueError(f"{value} is not an element of the interval [0,1]")

    return value
