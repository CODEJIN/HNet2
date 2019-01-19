from enum import Enum;

class Tensor_Type(Enum):
    Placeholder = 0,
    Tensor = 1,
    RNN_Tensor = 2,
    Create = 3,
    Loss = 4,
    Optimizer = 5,

class Learning_Rate_Decay_Method(Enum):
    No_Decay = 0,
    Exponential = 1,
    Noam = 2,

class Model_State(Enum):
    Paused = 0,
    Running = 1,
    Pausing = 2,
    Finished = 3,