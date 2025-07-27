from dataclasses import dataclass

@dataclass
class RPQN_config:
    mem: int = 3 # Memory size for the quasi-Newton method
    method: str = "BFGS" # Method to use, e.g., "BFGS", "SR1", "PSB", "DFP"
    c1: float = 0.1 # 0 < c1 < 0.5
    c2: float = 0.9 # c1 < c2 < 1
    sgm1: float = 0.5 # 0 < sgm1 < 1
    sgm2: float = 4 # sgm2 > 1
    crit_stop: float = 1e-6
    num_iter: int = 200