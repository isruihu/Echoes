"""
Author: Rui Hu
All rights reserved.
"""

from .erm import ERMTrainer
from .echoes import EchoesTrainer
from .lff import LfFTrainer
from .debian import DebiANTrainer
from .bpa import BPATrainer
from .jtt import JTTTrainer

methods = {
    'erm': ERMTrainer,
    'echoes': EchoesTrainer,
    'lff': LfFTrainer,
    'debian': DebiANTrainer,
    'bpa': BPATrainer,
    'jtt': JTTTrainer
}