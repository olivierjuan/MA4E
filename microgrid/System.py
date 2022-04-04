from MicroGridEnv import MicroGridEnv


class System:
    def __init__(self, nb_pdt=24 * 4):
        self.env = MicroGridEnv(envs={}, nb_pdt=24 * 4)
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

