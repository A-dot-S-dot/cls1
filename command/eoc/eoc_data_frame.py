import pandas as pd

from .eoc_calculator import EOCCalculator


class EOCDataFrame:
    dofs_format = "{:.0f}"
    error_format = "{:.2e}"
    eoc_format = "{:.2f}"

    _eoc_calculator: EOCCalculator
    _data_frame: pd.DataFrame

    def __init__(self, eoc_calculator: EOCCalculator):
        self._eoc_calculator = eoc_calculator
        self._create_empty_data_frame()
        self._fill_data_frame()
        self._format_data_frame()

    def _create_empty_data_frame(self):
        columns = pd.MultiIndex.from_product(
            [self._eoc_calculator.norm_names, ["error", "eoc"]]
        )
        columns = pd.MultiIndex.from_tuples([("DOFs", ""), *columns])
        index = pd.Index(
            [i for i in range(self._eoc_calculator.refine_number + 1)],
            name="refinement",
        )

        self._data_frame = pd.DataFrame(columns=columns, index=index)

    def _fill_data_frame(self):
        self._data_frame["DOFs"] = self._eoc_calculator.dofs

        for norm_index, norm in enumerate(self._eoc_calculator.norm_names):
            self._data_frame[norm, "error"] = self._eoc_calculator.errors[norm_index]
            self._data_frame[norm, "eoc"] = self._eoc_calculator.eocs[norm_index]

    def _format_data_frame(self):
        self._data_frame["DOFs"] = self._data_frame["DOFs"].apply(
            self.dofs_format.format
        )

        for norm in self._eoc_calculator.norm_names:
            self._data_frame[norm, "error"] = self._data_frame[norm, "error"].apply(
                self.error_format.format
            )
            self._data_frame[norm, "eoc"] = self._data_frame[norm, "eoc"].apply(
                self.eoc_format.format
            )

    def __repr__(self) -> str:
        return self._data_frame.__repr__()
