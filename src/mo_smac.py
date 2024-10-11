from smac.facade.multi_objective_facade import MultiObjectiveFacade
from smac.intensifier.intensifier import Intensifier
from smac import Scenario


class MOFacade(MultiObjectiveFacade):
    """
    Multi-objective facade for SMAC, set to keep track of 20 incumbents instead of the default 10.
    """
    
    @staticmethod
    def get_intensifier(  # type: ignore
            scenario: Scenario,
            *,
            max_config_calls: int = 2000,
            **kwargs,
    ) -> Intensifier:
        return super(MOFacade, MOFacade).get_intensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=20
        )

