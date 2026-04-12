from enum import Enum
from ..benchmark.bs_delta import BSDeltaBenchmark
from ..benchmark.sabr_practitioner_delta import SABRPractitionerDeltaBenchmark
from ..benchmark.bartlett_delta import BartlettDeltaBenchmark
from ..hedging_agents import DeepDPGHedgingAgent, SkewDeepDPGHedgingAgent
from ..simulation import GBMProcess, SABRProcess, SVJProcess


class ProcessType(Enum):
    GBM  = GBMProcess
    SABR = SABRProcess
    SVJ  = SVJProcess

class AgentType(Enum):
    DeepDPG   = DeepDPGHedgingAgent
    SkewDDPG   = SkewDeepDPGHedgingAgent

class BenchmarkType(Enum):
    BsDelta              = BSDeltaBenchmark
    SABRPractitionerDelta = SABRPractitionerDeltaBenchmark
    BartlettDelta        = BartlettDeltaBenchmark
