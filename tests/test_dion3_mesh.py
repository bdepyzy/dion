from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dion import Dion3
from test_megabatch_flatten import _gloo_all_to_all


def _worker(rank, ndim, rendezvous):
    torch.set_num_threads(1)
    torch.compiler.set_stance("force_eager")
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2**ndim,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh(
            "cpu", (2,) * ndim, mesh_dim_names=("dp", "fs", "tp")[-ndim:]
        )
        placements = (Replicate(),) * (ndim - 2) + (Shard(0), Shard(1))
        torch.manual_seed(42)
        weights = [torch.randn(7, 11) for _ in range(2)]
        params = [
            torch.nn.Parameter(distribute_tensor(w.clone(), mesh, placements))
            for w in weights
        ]
        refs = [
            torch.nn.Parameter(distribute_tensor(w.clone(), mesh["fs"], [Shard(0)]))
            for w in weights
        ]
        opt = Dion3(params, distributed_mesh=mesh, lr=0.02, fraction=0.5)
        ref_opt = Dion3(refs, distributed_mesh=mesh["fs"], lr=0.02, fraction=0.5)
        with patch.object(dist, "all_to_all", _gloo_all_to_all):
            for _ in range(3):
                for p, r in zip(params, refs):
                    grad = torch.randn(7, 11)
                    p.grad = distribute_tensor(grad.clone(), mesh, placements)
                    r.grad = distribute_tensor(grad.clone(), mesh["fs"], [Shard(0)])
                opt.step()
                ref_opt.step()
                for p, r in zip(params, refs):
                    assert p.placements == placements
                    for actual, expected in [(p, r)] + [
                        (opt.state[p][key], ref_opt.state[r][key])
                        for key in ("momentum", "variance_neuron")
                    ]:
                        torch.testing.assert_close(
                            actual.full_tensor(),
                            expected.full_tensor(),
                            rtol=1e-5,
                            atol=1e-6,
                        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ndim", [2, 3])
def test_dion3_mesh_matches_1d(ndim, tmp_path):
    mp.spawn(
        _worker,
        args=(ndim, (tmp_path / "rdzv").as_uri()),
        nprocs=2**ndim,
        join=True,
    )
