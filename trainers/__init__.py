"""
Author: Rui Hu
All rights reserved.
"""

from .erm import ERMTrainer
from .jtt import JTTTrainer
from .gce import GCETrainer
from .groupdro import GroupDROTrainer
from .lff import LfFTrainer
from .debian import DebiANTrainer
from .bpa import BPATrainer
from .echoes_b import EchoesBiasedModelTrainer
from .echoes import EchoesTrainer

methods = {
    'erm': ERMTrainer,
    'jtt': JTTTrainer,
    'gce': GCETrainer,
    'groupdro': GroupDROTrainer,
    'lff': LfFTrainer,
    'debian': DebiANTrainer,
    'bpa': BPATrainer,
    'echoes_b': EchoesBiasedModelTrainer,
    'echoes': EchoesTrainer
}