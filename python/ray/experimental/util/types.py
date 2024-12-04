from enum import Enum

from ray.util.annotations import PublicAPI


class _CollectiveOp(Enum):
    pass


@PublicAPI
class ReduceOp(_CollectiveOp):
    SUM = 0
    PRODUCT = 1
    MAX = 2
    MIN = 3
    AVG = 4

    def __str__(self):
        return f"{self.name.lower()}"

@PublicAPI
class AllGatherOp(_CollectiveOp):
    # Single operation type since allgather just concatenates tensors
    CONCAT = 0

    def __str__(self):
        return "allgather"