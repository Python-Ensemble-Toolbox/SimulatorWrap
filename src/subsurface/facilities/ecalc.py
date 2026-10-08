"""eCalc: the facility's CO2 emissions, added to the output of a reservoir simulator."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

__all__ = ['Ecalc', 'EcalcModel']


class Ecalc:
    """A reservoir simulator followed by eCalc, run together for every member.

    The reservoir simulator runs as usual, then eCalc computes the facility's CO2
    emissions from that member's production, and the emission rate is added to the
    member's output as ``FU_CO2R`` [ton/day]: entry k is the rate between report
    dates k-1 and k, entry 0 is zero. ``subsurface.cost_functions.npv`` prices it at
    ``wem`` USD/ton.

    eCalc runs inside each member's simulation, so it runs in parallel like the
    simulations do. Everything not defined here is taken from the reservoir
    simulator, so the wrapper can be used wherever the simulator itself is. The HPC
    queue reads the simulator's files directly and skips eCalc.

    Parameters
    ----------
    reservoir : simulator
        The reservoir simulator, e.g. ``subsurface.multphaseflow.opm.flow``. Its
        report points must be dates, and its ``datatype`` must include the cumulative
        field volumes FOPT, FGPT, FWPT and FWIT.
    ecalc_config : str or Path
        The eCalc model file, see :class:`EcalcModel`.
    """

    OUTPUTS = ['FU_CO2R']

    def __init__(self, reservoir: Any, ecalc_config: str | Path) -> None:
        if reservoir.true_order[0] != 'dates':
            raise ValueError(f"eCalc needs report dates, not '{reservoir.true_order[0]}'.")
        self.reservoir = reservoir
        self.model = EcalcModel(ecalc_config)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on the wrapper itself
        if name == 'reservoir':
            raise AttributeError(name)
        return getattr(self.reservoir, name)

    @property
    def redund_sim(self) -> Any:
        return self.reservoir.redund_sim

    @redund_sim.setter
    def redund_sim(self, sim: Any) -> None:
        # The backup simulator replaces the reservoir simulator when it fails, so it belongs there
        self.reservoir.redund_sim = sim

    @property
    def datatype(self) -> list[str]:
        return [*self.reservoir.input_dict['datatype'], *self.OUTPUTS]

    def setup_fwd_run(self, **kwargs: Any) -> None:
        self.reservoir.setup_fwd_run(**kwargs)

    def run_fwd_sim(self, state: dict, member_i: int, *args: Any, **kwargs: Any) -> list[dict] | bool:
        """Run the reservoir simulator for one member, then add the facility's CO2 emission rate."""
        output = self.reservoir.run_fwd_sim(state, member_i, *args, **kwargs)
        if output is False:
            return False

        cumulative = pd.DataFrame.from_records(output, index=pd.DatetimeIndex(self.reservoir.true_order[1]))
        for record, rate in zip(output, self.model.co2_rate(cumulative)):
            record['FU_CO2R'] = rate

        return output


class EcalcModel:
    """An eCalc model whose production time series is supplied in memory rather than read from file.

    The model file needs exactly one ``TIME_SERIES`` entry. The production is passed
    to it as the field rates FOPR, FGPR, FWPR and FWIR [Sm3/day], each the average over
    a report step (from the cumulative volumes), so the model refers to them as e.g.
    ``SIM1;FOPR``. The model's START and END are set to the first and last report
    dates. Every emission whose name contains "co2" is counted.
    """

    def __init__(self, config_file: str | Path) -> None:
        self.path = Path(config_file).resolve()
        self.config = yaml.safe_load(self.path.read_text())
        series = self.config.get('TIME_SERIES', [])
        if len(series) != 1:
            raise ValueError(f'{self.path.name} must have exactly one TIME_SERIES entry, it has {len(series)}.')
        self.series_file = series[0]['FILE']

    def co2_rate(self, cumulative: pd.DataFrame) -> np.ndarray:
        """CO2 emission rate [ton/day] in each report step, from cumulative field volumes indexed by date.

        Entry k is the rate between report dates k-1 and k; entry 0 is zero.
        """
        from ecalc_cli.infrastructure.file_resource_service import FileResourceService
        from ecalc_neqsim_wrapper import NeqsimService
        from libecalc.presentation.json_result.mapper import get_asset_result
        from libecalc.presentation.yaml.domain.time_series_resource import TimeSeriesResource
        from libecalc.presentation.yaml.model import YamlModel
        from libecalc.presentation.yaml.yaml_entities import MemoryResource, ResourceStream
        from libecalc.presentation.yaml.yaml_models.yaml_model import ReaderType, YamlConfiguration

        # Average rate in each report step, dated at the start of the step (eCalc holds a value until the next date)
        dates = pd.DatetimeIndex(cumulative.index)
        days = np.diff(dates).astype('timedelta64[D]').astype(float)
        rates = {
            rate: np.diff(cumulative[total].to_numpy(dtype=float)) / days
            for rate, total in [('FOPR', 'FOPT'), ('FGPR', 'FGPT'), ('FWPR', 'FWPT'), ('FWIR', 'FWIT')]
        }
        series = MemoryResource(
            headers=['DATE', *rates],
            data=[[d.strftime('%Y-%m-%d') for d in dates[:-1]], *[list(r) for r in rates.values()]],
        )

        # The model, run over the report period
        config = dict(self.config, START=dates[0].strftime('%Y-%m-%d'), END=dates[-1].strftime('%Y-%m-%d'))
        stream = ResourceStream(name=self.path.stem, stream=io.StringIO(yaml.safe_dump(config, sort_keys=False)))
        configuration = YamlConfiguration.Builder.get_yaml_reader(ReaderType.PYYAML).get_validator(
            main_yaml=stream, base_dir=self.path.parent, enable_include=True,
        )
        series_file = self.series_file

        class Resources(FileResourceService):
            """Facility inputs from file, the production time series from memory."""
            def get_time_series_resources(self) -> tuple[dict[str, TimeSeriesResource], list]:
                return {series_file: TimeSeriesResource(series).validate()}, []

        with NeqsimService.factory().initialize():
            model = YamlModel(
                configuration=configuration,
                resource_service=Resources(working_directory=self.path.parent, configuration=configuration),
            ).validate_for_run()
            model.evaluate_energy_usage()
            emissions = get_asset_result(model).component_result.emissions

        co2 = sum(np.asarray(e.rate.values) for name, e in emissions.items() if 'co2' in name.lower())
        return np.concatenate([[0.0], co2])
