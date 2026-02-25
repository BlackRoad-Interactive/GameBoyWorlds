from typing import Optional, Union, Type, Dict
from poke_worlds.emulation.parser import StateParser
from poke_worlds.emulation.tracker import StateTracker
from poke_worlds.emulation.emulator import Emulator

from poke_worlds.emulation.pokemon.parsers import (
    MemoryBasedPokemonRedStateParser,
    PokemonBrownStateParser,
    PokemonStarBeastsStateParser,
    PokemonCrystalStateParser,
    PokemonPrismStateParser,
    PokemonFoolsGoldStateParser,
)
from poke_worlds.emulation.pokemon.trackers import (
    PokemonOCRTracker,
    PokemonRedStarterTracker,
    PokemonRedCenterTestTracker,
    PokemonRedMtMoonTestTracker,
    PokemonRedSpeakToBillTestTracker,
    PokemonRedPickupPokeballTestTracker,
    PokemonRedSpeakToBillTestTracker,
    PokemonRedPickupPokeballTestTracker,
    PokemonRedReadTrainersTipsSignTestTracker,
    PokemonRedSpeakToCinnabarGymAideCompleteTestTracker,
    PokemonRedSpeakToCinnabarMonkTestTracker,
    PokemonRedDefeatedBrockTestTracker,
    PokemonRedDefeatedLassTestTracker,
    PokemonRedCaughtPidgeyTestTracker,
    PokemonRedCaughtPikachuTestTracker,
    PokemonRedBoughtPotionAtPewterPokemartTestTracker,
    PokemonRedUsedPotionOnCharmanderTestTracker,
    PokemonRedOpenMapTestTracker,
)
from poke_worlds.emulation.pokemon.emulators import PokemonEmulator


GAME_TO_GB_NAME = {
    "pokemon_red": "PokemonRed.gb",
    "pokemon_brown": "PokemonBrown.gb",
    "pokemon_starbeasts": "PokemonStarBeasts.gb",
    "pokemon_crystal": "PokemonCrystal.gbc",
    "pokemon_fools_gold": "PokemonFoolsGold.gbc",
    "pokemon_prism": "PokemonPrism.gbc",
}
""" Expected save name for each game. Save the file to <storage_dir_from_config_file>/<game_name>_rom_data/<gb_name>"""

STRONGEST_PARSERS: Dict[str, Type[StateParser]] = {
    "pokemon_red": MemoryBasedPokemonRedStateParser,
    "pokemon_brown": PokemonBrownStateParser,
    "pokemon_crystal": PokemonCrystalStateParser,
    "pokemon_starbeasts": PokemonStarBeastsStateParser,
    "pokemon_fools_gold": PokemonFoolsGoldStateParser,
    "pokemon_prism": PokemonPrismStateParser,
}
""" Mapping of game names to their corresponding strongest StateParser classes. 
Unless you have a very good reason, you should always use the STRONGEST possible parser for a given game. 
The parser itself does not affect performance, as for it to perform a read / screen comparison operation , it must be called upon by the state tracker.
This means there is never a reason to use a weaker parser. 
"""


AVAILABLE_STATE_TRACKERS: Dict[str, Dict[str, Type[StateTracker]]] = {
    "pokemon_red": {
        "default": PokemonOCRTracker,
        "starter_example": PokemonRedStarterTracker,
        "viridian_center_test": PokemonRedCenterTestTracker,
        "mt_moon_test": PokemonRedMtMoonTestTracker,
        "speak_to_bill_test": PokemonRedSpeakToBillTestTracker,
        "pickup_pokeball_test": PokemonRedPickupPokeballTestTracker,
        "read_trainers_tips_sign_test": PokemonRedReadTrainersTipsSignTestTracker,
        "speak_to_cinnabar_gym_aide_complete_test": PokemonRedSpeakToCinnabarGymAideCompleteTestTracker,
        "speak_to_cinnabar_monk_test": PokemonRedSpeakToCinnabarMonkTestTracker,
        "defeated_brock_test": PokemonRedDefeatedBrockTestTracker,
        "defeated_lass_test": PokemonRedDefeatedLassTestTracker,
        "caught_pidgey_test": PokemonRedCaughtPidgeyTestTracker,
        "caught_pikachu_test": PokemonRedCaughtPikachuTestTracker,
        "bought_potion_at_pewter_pokemart_test": PokemonRedBoughtPotionAtPewterPokemartTestTracker,
        "used_potion_on_charmander_test": PokemonRedUsedPotionOnCharmanderTestTracker,
        "open_map_test": PokemonRedOpenMapTestTracker,
    },
    "pokemon_brown": {
        "default": PokemonOCRTracker,
    },
    "pokemon_crystal": {
        "default": PokemonOCRTracker,
    },
    "pokemon_starbeasts": {
        "default": PokemonOCRTracker,
    },
    "pokemon_fools_gold": {
        "default": PokemonOCRTracker,
    },
    "pokemon_prism": {
        "default": PokemonOCRTracker,
    },
}
""" Mapping of game names to their available StateTracker classes with string identifiers. """


AVAILABLE_EMULATORS: Dict[str, Dict[str, Type[Emulator]]] = {
    "pokemon_red": {
        "default": PokemonEmulator,
    },
    "pokemon_brown": {
        "default": PokemonEmulator,
    },
    "pokemon_crystal": {
        "default": PokemonEmulator,
    },
    "pokemon_starbeasts": {
        "default": PokemonEmulator,
    },
    "pokemon_fools_gold": {
        "default": PokemonEmulator,
    },
    "pokemon_prism": {
        "default": PokemonEmulator,
    },
}
""" Mapping of game names to their available Emulator classes with string identifiers. """
