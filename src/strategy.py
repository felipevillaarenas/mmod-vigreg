from contextlib import nullcontext

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase


class CacheDDPStrategy(DDPStrategy):

    def configure_ddp(self) -> None:
        """
        Custom DDP Strategy that clean chache in the cuda device 
        before seting-up the model.
        """        
        self.model = self._setup_model(_LightningModuleWrapperBase(self.model))
        self._register_ddp_hooks()

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
        with ctx:
            rank = self.global_rank
            print(f"Start running basic DDP example on rank {rank}.")
            print("device_ids", device_ids)
            # create model and move it to GPU with id rank
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            model = model.to(device_id)
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            return DistributedDataParallel(module=model, device_ids=[device_id], **self._ddp_kwargs)