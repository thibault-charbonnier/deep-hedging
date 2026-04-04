from enum import Enum
from ..benchmark import BSDeltaBenchmark
from ..simulation import GBMProcess, SABRProcess, SVJProcess
from ..hedging_agents import DQNHedgingAgent, DoubleQDNHedgingAgent, DeepDPGHedgingAgent


class ProcessType(Enum):
    GBM = GBMProcess
    SABR = SABRProcess
    SVJ = SVJProcess

class AgentType(Enum):
    DQN = DQNHedgingAgent
    DoubleDQN = DoubleQDNHedgingAgent
    DeepDPG = DeepDPGHedgingAgent

class BenchmarkType(Enum):
    BsDelta = BSDeltaBenchmark