import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from .linear import linear
from .embedding import embedding
from .activations import *
from .bpe import train_bpe
from .rope import *
from .attention import *
from .rmsnorm import *
from .transformer import *
from .gradients import *
from .loss import *
from .optimizers import *
from .schedulers import *
from .tokenizer import *
from .utils import *