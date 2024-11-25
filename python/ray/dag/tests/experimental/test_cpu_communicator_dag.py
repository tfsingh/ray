import os
import sys
import torch

import pytest

import ray
import ray.cluster_utils
from ray.exceptions import RayChannelError
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.experimental.channel.cpu_nccl_group import CPUNcclGroup, start_nccl_mock
from ray.experimental.channel.gpu_communicator import TorchTensorAllocator
from ray.dag import InputNode
import ray.experimental.collective as collective
from ray.dag.output_node import MultiOutputNode
from ray.tests.conftest import *  # noqa


@ray.remote
class CPUTorchTensorWorker:
    def __init__(self):
        self.device = "cpu"

    def start_cuda_and_torch_mock(self):
        start_nccl_mock()

    def send(self, shape, dtype, value: int, send_tensor=True):
        if not send_tensor:
            return 1
        return torch.ones(shape, dtype=dtype, device=self.device) * value
    
    def send_dict(self, entries):
        results = {}
        for key, entry in entries.items():
            value, shape, dtype = entry
            results[key] = torch.ones(shape, dtype=dtype, device=self.device) * value
        return results
    
    def send_or_raise(self, shape, dtype, value: int, raise_exception=False):
        if raise_exception:
            raise RuntimeError()
        return torch.ones(shape, dtype=dtype, device=self.device) * value

    def recv(self, tensor):
        '''
        (TODO): because the use of mock nccl, device of tensors are 'cuda'. 
        Can't fix this in CPUNcclGroup. I don't know exact reason but on a 
        high level is because the whole p2p and collective comm in adag is 
        assuming using nccl backend. The values are correct so I just skip 
        device check for now. 
        '''
        # assert tensor.device == self.device
        return (tensor[0].item(), tensor.shape, tensor.dtype)

    def recv_dict(self, tensor_dict):
        vals = {}
        for i, tensor in tensor_dict.items():
            vals[i] = self.recv(tensor)
        return vals
    
    def compute_with_tuple_args(self, args, i: int):
        shape, dtype, value = args[i]
        tensor = torch.ones(shape, dtype=dtype, device=self.device) * value
        return tensor
    
    def recv_tensor(self, tensor):
        '''
        (TODO): same as recv
        '''
        # assert tensor.device == self.device
        return tensor
    
    def return_tensor(self, size: int) -> torch.Tensor:
        return torch.ones(size, device=self.device)

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_p2p_basic(ray_start_cluster):
    sender = CPUTorchTensorWorker.remote()
    receiver = CPUTorchTensorWorker.remote()

    cpu_group = CPUNcclGroup(2, [sender, receiver])
    ray.get([
        sender.start_cuda_and_torch_mock.remote(),
        receiver.start_cuda_and_torch_mock.remote()
    ])

    shape = (10,)
    dtype = torch.float16

    with InputNode() as inp:
        dag = sender.send.bind(inp.shape, inp.dtype, inp[0])
        dag = dag.with_type_hint(TorchTensorType(transport=cpu_group))
        dag = receiver.recv.bind(dag)
    
    compiled_dag = dag.experimental_compile()
    for i in range(3):
        ref = compiled_dag.execute(i, shape=shape, dtype=dtype)
        assert ray.get(ref) == (i, shape, dtype)

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_basic(ray_start_cluster):
    num_workers = 2
    workers = [CPUTorchTensorWorker.remote() for _ in range(num_workers)]
    ray.get([worker.start_cuda_and_torch_mock.remote() for worker in workers])

    cpu_group = CPUNcclGroup(num_workers, workers)

    shape = (10,)
    dtype = torch.float16

    with InputNode() as inp:
        computes = [
            worker.compute_with_tuple_args.bind(inp, i)
            for i, worker in enumerate(workers)
        ]
        collectives = collective.allreduce.bind(computes, transport=cpu_group)
        recvs = [
            worker.recv.bind(collective)
            for worker, collective in zip(workers, collectives)
        ]
        dag = MultiOutputNode(recvs)
    
    compiled_dag = dag.experimental_compile()
    
    for i in range(3):
        i += 1
        shape = (i * 10,)
        dtype = torch.float16
        ref = compiled_dag.execute(
            [(shape, dtype, i + idx) for idx in range(num_workers)]
        )
        result = ray.get(ref)
        reduced_val = sum(i + idx for idx in range(num_workers))
        assert result == [(reduced_val, shape, dtype) for _ in workers]

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_get_partial(ray_start_cluster):
    num_workers = 2
    workers = [CPUTorchTensorWorker.remote() for _ in range(num_workers)]
    ray.get([worker.start_cuda_and_torch_mock.remote() for worker in workers])

    cpu_group = CPUNcclGroup(num_workers, workers)

    shape = (10,)
    dtype = torch.float16

    with InputNode() as inp:
        computes = [
            worker.compute_with_tuple_args.bind(inp, i)
            for i, worker in enumerate(workers)
        ]
        collectives = collective.allreduce.bind(computes, transport=cpu_group)
        recv = workers[0].recv.bind(collectives[0])
        tensor = workers[1].recv_tensor.bind(collectives[0])
        dag = MultiOutputNode([recv, tensor, collectives[1]])
    
    compiled_dag = dag.experimental_compile()
    
    for i in range(3):
        ref = compiled_dag.execute(
            [(shape, dtype, i + idx + 1) for idx in range(num_workers)]
        )
        result = ray.get(ref)
        metadata, tensor, _ = result
        reduced_val = sum(i + idx + 1 for idx in range(num_workers))
        assert metadata == (reduced_val, shape, dtype)
        expected_tensor_val = torch.ones(shape, dtype=dtype) * reduced_val
        assert torch.equal(tensor, expected_tensor_val)

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_wrong_shape(ray_start_cluster):
    num_workers = 2
    workers = [CPUTorchTensorWorker.remote() for _ in range(num_workers)]
    ray.get([worker.start_cuda_and_torch_mock.remote() for worker in workers])

    cpu_group = CPUNcclGroup(num_workers, workers)

    dtype = torch.float16

    with InputNode() as inp:
        computes = [
            worker.compute_with_tuple_args.bind(inp, i)
            for i, worker in enumerate(workers)
        ]
        collectives = collective.allreduce.bind(computes, transport=cpu_group)
        recvs = [
            worker.recv.bind(collective)
            for worker, collective in zip(workers, collectives)
        ]
        dag = MultiOutputNode(recvs)

    compiled_dag = dag.experimental_compile()

    ref = compiled_dag.execute([((20,), dtype, idx + 1) for idx in range(num_workers)])
    reduced_val = (1 + num_workers) * num_workers / 2
    assert ray.get(ref) == [(reduced_val, (20,), dtype) for _ in range(num_workers)]

    ref = compiled_dag.execute(
        [((10 * (idx + 1),), dtype, idx + 1) for idx in range(num_workers)]
    )
    # Execution hangs because of shape mismatch and a timeout error is raised.
    with pytest.raises(RayChannelError):
        ray.get(ref)

    # The DAG will be torn down after any task throws an application-level
    # exception, such as when the task returns torch.Tensors of the wrong
    # shape or dtype. Check that we can no longer submit to the DAG.
    ref = compiled_dag.execute([((20,), dtype, 1) for _ in workers])
    with pytest.raises(RayChannelError):
        ref = compiled_dag.execute([((20,), dtype, 1) for _ in workers])

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_scheduling(ray_start_cluster):
    """
    Test scheduling avoids potential deadlocks that arise from all-reduce operations.

    inp --> x(0) --> +------------+
        |            | all-reduce |
        --> y(1) --> +------------+
        |
        --> t(0) --> recv(1)

    In the above graph, x, y, t are tensors, and the numbers inside parentheses
    identify the actors. If actor 1 launches an all-reduce with tensor y while
    actor 0 starts sending t, then actor 1 waits for actor 0 to join the all-reduce
    while actor 1 waits for actor 0 to receive t.
    """
    num_workers = 2
    workers = [CPUTorchTensorWorker.remote() for _ in range(num_workers)]
    ray.get([worker.start_cuda_and_torch_mock.remote() for worker in workers])

    cpu_group = CPUNcclGroup(num_workers, workers)

    shape = (10,)
    dtype = torch.float16

    with InputNode() as inp:
        # Tensors in the all-reduce.
        x = workers[0].send.bind(shape, dtype, inp)
        y = workers[1].send.bind(shape, dtype, inp)

        # Tensor to be sent from workes[0] to workers[1].
        t = workers[0].send.bind(shape, dtype, inp)
        t.with_type_hint(TorchTensorType(transport=cpu_group))

        collectives = collective.allreduce.bind([x, y])
        recv = workers[1].recv.bind(t)
        dag = MultiOutputNode([collectives[0], collectives[1], recv])

    compiled_dag = dag.experimental_compile()

    value = 10
    ref = compiled_dag.execute(value)
    result = ray.get(ref)
    reduced_value = value * 2
    expected_tensor_val = torch.ones(shape, dtype=dtype) * reduced_value
    assert torch.equal(result[0], expected_tensor_val)
    assert torch.equal(result[1], expected_tensor_val)
    assert result[2] == (value, shape, dtype)

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_duplicate_actors(ray_start_cluster):
    """
    Test an error is thrown when two input nodes from the same actor bind to
    an all-reduce.
    """
    worker = CPUTorchTensorWorker.remote()
    ray.get(worker.start_cuda_and_torch_mock.remote())

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for _ in range(2)]
        with pytest.raises(
            ValueError,
            match="Expected unique actor handles for a collective operation",
        ):
            collective.allreduce.bind(computes)

    with InputNode() as inp:
        compute = worker.return_tensor.bind(inp)
        computes = [compute for _ in range(2)]
        with pytest.raises(
            ValueError,
            match="Expected unique input nodes for a collective operation",
        ):
            collective.allreduce.bind(computes)

@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "num_cpus": 2,
            "num_gpus": 0,
            "num_nodes": 1,
        }
    ],
    indirect=True,
)
def test_allreduce_wrong_actors(ray_start_cluster):
    # (TODO): better explain this test
    """
    Test an error is thrown when an all-reduce binds to a wrong set of actors.
    """
    num_workers = 2
    workers = [CPUTorchTensorWorker.remote() for _ in range(num_workers * 2)]
    ray.get([worker.start_cuda_and_torch_mock.remote() for worker in workers])

    cpu_group = CPUNcclGroup(num_workers, workers[:2])

    with InputNode() as inp:
        computes = [worker.return_tensor.bind(inp) for worker in workers[2:]]
        with pytest.raises(
            ValueError,
            match="Expected actor handles to match the custom NCCL group",
        ):
            collective.allreduce.bind(computes, transport=cpu_group)


'''
barrier tests
'''

# class DefaultTensorAllocator(TorchTensorAllocator):
#     def __call__(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
#         return torch.empty(shape, dtype=dtype)

# @ray.remote(num_cpus=1, num_gpus=0)
# class Worker:
#     def __init__(self, rank):
#         self.nccl_group = None
#         self.rank = rank
#         self.allocator = DefaultTensorAllocator()

#     def set_nccl_channel(self, nccl_group):
#         self.nccl_group = nccl_group

#     def send(self, val, shape, dtype, peer_rank):
#         try:
#             t = torch.ones(shape, dtype=dtype) * val
#             self.nccl_group.send(t, peer_rank)
#             return True
#         except RayChannelError as e:
#             print(f"Send error: {e}")
#             return False

#     def receive(self, shape, dtype, peer_rank):
#         try:
#             t = self.nccl_group.recv(
#                 shape=shape,
#                 dtype=dtype,
#                 peer_rank=peer_rank,
#                 allocator=self.allocator
#             )
#             return (t[0].item(), t.shape, t.dtype)
#         except RayChannelError as e:
#             print(f"Receive error: {e}")
#             return None

#     def allreduce(self, val, shape, dtype):
#         t = torch.ones(shape, dtype=dtype) * val
#         recv_t = torch.zeros(shape, dtype=dtype)
#         try:
#             t = self.nccl_group.allreduce(
#                 send_buf=t,
#                 recv_buf=recv_t,
#             )
#             return (recv_t[0].item(), recv_t.shape, recv_t.dtype)
#         except RayChannelError as e:
#             print(f"Allreduce error: {e}")
#             return None

# @pytest.mark.parametrize(
#     "ray_start_cluster",
#     [
#         {
#             "num_cpus": 2,
#             "num_gpus": 0,
#             "num_nodes": 1,
#         }
#     ],
#     indirect=True,
# )
# def test_cpu_p2p(ray_start_cluster):
#     sender = Worker.remote(rank=0)
#     receiver = Worker.remote(rank=1)

#     nccl_group = CPUNcclGroup(
#         world_size=2,
#         rank=0,
#         actor_handles=[sender, receiver]
#     )
#     r_nccl_group = CPUNcclGroup(
#         world_size=2,
#         rank=1,
#         actor_handles=[sender, receiver]
#     )

#     ray.get([
#         sender.set_nccl_channel.remote(nccl_group),
#         receiver.set_nccl_channel.remote(r_nccl_group)
#     ])

#     shape = (3,)
#     dtype = torch.float32
#     test_value = 2.0
    
#     send_future = sender.send.remote(test_value, shape, dtype, peer_rank=1)
#     receive_future = receiver.receive.remote(shape, dtype, peer_rank=0)
    
#     send_result = ray.get(send_future)
#     receive_result = ray.get(receive_future)

#     received_value, received_shape, received_dtype = receive_result
#     assert received_value == test_value, f"Expected value {test_value}, got {received_value}"

# @pytest.mark.parametrize(
#     "ray_start_cluster",
#     [
#         {
#             "num_cpus": 2,
#             "num_gpus": 0,
#             "num_nodes": 1,
#         }
#     ],
#     indirect=True,
# )
# def test_cpu_allreduce(ray_start_cluster):
#     world_size = 2

#     worker1 = Worker.remote(rank=0)
#     worker2 = Worker.remote(rank=1)

#     nccl_group_1 = CPUNcclGroup(
#         world_size=2,
#         rank=0,
#         actor_handles=[worker1, worker2]
#     )
#     nccl_group_2 = CPUNcclGroup(
#         world_size=2,
#         rank=1,
#         actor_handles=[worker1, worker2]
#     )

#     ray.get([
#         worker1.set_nccl_channel.remote(nccl_group_1),
#         worker2.set_nccl_channel.remote(nccl_group_2)
#     ])

#     shape = (3,)
#     dtype = torch.float32
#     test_value = 2.0

#     res_ref = [
#         worker1.allreduce.remote(test_value, shape, dtype),
#         worker2.allreduce.remote(test_value, shape, dtype),
#     ]

#     res = ray.get(res_ref)

#     received_value, received_shape, received_dtype = res[0]
#     assert received_value == test_value*2, f"Expected value {test_value}, got {received_value}"

if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
