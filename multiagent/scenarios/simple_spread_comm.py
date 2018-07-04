import itertools as it

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenarios.simple_spread import Scenario as NoCommScenario

class Scenario(NoCommScenario):
    def make_world(self):
        world = super().make_world()
        for agent in world.agents:
            agent.silent = False
        return world
