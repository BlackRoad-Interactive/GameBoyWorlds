from typing import Dict, Type
from poke_worlds.interface.controller import Controller
from poke_worlds.interface.environment import Environment, DummyEnvironment
from poke_worlds.interface.pokemon.environments import (
    PokemonEnvironment,
    PokemonRedChooseCharmanderEnvironment,
    PokemonRedChooseCharmanderEasyEnvironment,
    PokemonRedChooseCharmanderHardEnvironment,
    PokemonOCREnvironment,
    PokemonRedExploreStartingSceneEnvironment,
    PokemonTestEnvironment,
    PokemonTrainEnvironment,
)
from poke_worlds.interface.pokemon.controllers import PokemonStateWiseController

AVAILABLE_ENVIRONMENTS: Dict[str, Dict[str, Type[Environment]]] = {
    "pokemon_red": {
        "dummy": DummyEnvironment,
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "charmander": PokemonRedChooseCharmanderEnvironment,
        "charmander_easy": PokemonRedChooseCharmanderEasyEnvironment,
        "charmander_hard": PokemonRedChooseCharmanderHardEnvironment,
        "starter_explore": PokemonRedExploreStartingSceneEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
    "pokemon_brown": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
    "pokemon_starbeasts": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
    "pokemon_crystal": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
    "pokemon_prism": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
    "pokemon_fools_gold": {
        "default": PokemonOCREnvironment,
        "basic": PokemonEnvironment,
        "train": PokemonTrainEnvironment,
        "test": PokemonTestEnvironment,
    },
}

AVAILABLE_CONTROLLERS: Dict[str, Dict[str, Type[Controller]]] = {
    "pokemon_red": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_brown": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_crystal": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_starbeasts": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_prism": {
        "state_wise": PokemonStateWiseController,
    },
    "pokemon_fools_gold": {
        "state_wise": PokemonStateWiseController,
    },
}
