import logging
from typing import List, Optional, Union

import ray
from ray.dag.collective_node import CollectiveOutputNode, _CollectiveOperation
from ray.dag.constants import (
    BIND_INDEX_KEY,
    COLLECTIVE_OPERATION_KEY,
    PARENT_CLASS_NODE_KEY,
)
from ray.experimental.channel.torch_tensor_type import GPUCommunicator, TorchTensorType
from ray.experimental.util.types import AllGatherOp

logger = logging.getLogger(__name__)

class AllGatherWrapper:
    def bind(
        self,
        input_nodes: List["ray.dag.DAGNode"],
        transport: Optional[Union[str, GPUCommunicator]] = None,
    ) -> List[CollectiveOutputNode]:
        """
        Bind input nodes with an allgather operation.

        Args:
            input_nodes: A list of DAG nodes that each return a torch tensor
            transport: GPU communicator for the collective operation
                      If not specified, default NCCL is used
        Returns:
            List of collective output nodes containing gathered tensors
        """
        if transport is None:
            transport = TorchTensorType.NCCL

        collective_op = _CollectiveOperation(
            input_nodes,
            AllGatherOp.CONCAT,
            transport
        )
        collective_output_nodes = []

        for input_node in input_nodes:
            actor_handle = input_node._get_actor_handle()
            if actor_handle is None:
                raise ValueError("Expected an actor handle from the input node")

            collective_output_node = CollectiveOutputNode(
                method_name="allgather",
                method_args=(input_node,),
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve={
                    PARENT_CLASS_NODE_KEY: actor_handle,
                    BIND_INDEX_KEY: actor_handle._ray_dag_bind_index,
                    COLLECTIVE_OPERATION_KEY: collective_op,
                },
            )
            actor_handle._ray_dag_bind_index += 1
            collective_output_nodes.append(collective_output_node)

        return collective_output_nodes

    def __call__(self, tensor, group_name: str = "default"):
        from ray.util.collective.collective import allgather
        return allgather(tensor, group_name)

allgather = AllGatherWrapper()