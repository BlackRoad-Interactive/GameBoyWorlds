from typing import Dict, Type
from poke_worlds.interface.controller import Controller
from poke_worlds.interface.environment import Environment, DummyEnvironment


AVAILABLE_ENVIRONMENTS: Dict[str, Dict[str, Type[Environment]]] = {}

AVAILABLE_CONTROLLERS: Dict[str, Dict[str, Type[Controller]]] = {}
