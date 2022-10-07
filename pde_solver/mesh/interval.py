class Interval:
    """Represents an interval object."""

    _a: float
    _b: float

    def __init__(self, a, b):
        self._a = float(a)
        self._b = float(b)

        assert self._a < self._b, f"{a} is not strictly less than {b}"

    def __str__(self) -> str:
        return f"[{self.a}, {self.b}]"

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def length(self) -> float:
        return self._b - self._a

    def __hash__(self) -> int:
        return hash((self._a, self._b))

    def __eq__(self, other) -> bool:
        return self.a == other.a and self.b == other.b

    def __contains__(self, x: float) -> bool:
        return x >= self._a and x <= self._b

    def is_in_inner(self, x: float) -> bool:
        return x > self._a and x < self._b

    def is_in_boundary(self, x: float) -> bool:
        return x == self._a or x == self._b
