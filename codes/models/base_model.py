import os
import torch
import torch.nn as nn
import uuid


class BaseModel():

    def __init__(self, opt, name):
        self.name = name
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.checkpoints_dir = os.path.join("checkpoints", name)
        self.checkpoint_latest = os.path.join(self.checkpoints_dir, "latest")

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def all_networks_state_dict(self):
        pass

    def networks_index(self):
        pass

    def load_pretrained(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    # helper printing function that can be used by subclasses
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    # helper saving function that can be used by subclasses
    def network_state_dict(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        return state_dict

    def save_checkpoint(self, epoch, current_step):
        checkpoint = {
            "epoch": epoch,
            "current_step": current_step,
            "networks": self.all_networks_state_dict(),
            "schedulers": [],
            "optimizers": []
        }
        for s in self.schedulers:
            checkpoint["schedulers"].append(s.state_dict())
        for o in self.optimizers:
            checkpoint["optimizers"].append(o.state_dict())
        # Ensure checkpoints dir exists
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        # Save the checkpoint
        path = os.path.join(self.checkpoints_dir, str(current_step))
        torch.save(checkpoint, path)
        # NOTE: Instead of deleting the existing latest symlink and then creating the new latest
        # we create the new one as a temporary file first and then replace the existing latest.
        # This way, if we crash, we still always have a valid latest symlink.
        tmp_latest_link = os.path.join(self.checkpoints_dir, ".tmp{}".format(uuid.uuid4().hex))
        os.symlink(str(current_step), tmp_latest_link)
        os.replace(tmp_latest_link, self.checkpoint_latest)

    def load_network_from_path(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def load_checkpoint(self, strict=True):
        if not os.path.isfile(self.checkpoint_latest):
            # Attempt to load pretrained only if we do not already have a checkpoint
            self.load_pretrained()
            print("Checkpoint does not exist yet, returning epoch 0 step 0")
            return (0, 0)
        checkpoint = torch.load(self.checkpoint_latest)
        for (net_name, net) in self.networks_index().items():
            if isinstance(net, nn.DataParallel):
                net = net.module
            net.load_state_dict(checkpoint["networks"][net_name], strict=strict)
        assert len(checkpoint["schedulers"]) == len(self.schedulers)
        assert len(checkpoint["optimizers"]) == len(self.optimizers)
        for i in range(len(self.schedulers)):
            self.schedulers[i].load_state_dict(checkpoint["schedulers"][i])
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict(checkpoint["optimizers"][i])
        return (checkpoint["epoch"], checkpoint["current_step"])
