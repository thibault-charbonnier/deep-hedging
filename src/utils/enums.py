from enum import Enum
from ..benchmark.bs_delta import BSDeltaBenchmark
from ..benchmark.sabr_practitioner_delta import SABRPractitionerDeltaBenchmark
from ..benchmark.bartlett_delta import BartlettDeltaBenchmark
from ..hedging_agents import DQNHedgingAgent, DoubleQDNHedgingAgent, DeepDPGHedgingAgent
from ..simulation import GBMProcess, SABRProcess, SVJProcess


class ProcessType(Enum):
    GBM  = GBMProcess
    SABR = SABRProcess
    SVJ  = SVJProcess

class AgentType(Enum):
    DQN       = DQNHedgingAgent
    DoubleDQN = DoubleQDNHedgingAgent
    DeepDPG   = DeepDPGHedgingAgent

class BenchmarkType(Enum):
    BsDelta              = BSDeltaBenchmark
    SABRPractitionerDelta = SABRPractitionerDeltaBenchmark
    BartlettDelta        = BartlettDeltaBenchmark
