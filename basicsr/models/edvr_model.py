import torch

from basicsr.models.video_base_model import VideoBaseModel


class EDVRModel(VideoBaseModel):
    """EDVR Model.
    """

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if train_opt['tsa_iter']:
            normal_params = []
            tsa_fusion_params = []
            for k, v in self.net_g.named_parameters():
                if 'fusion' in k:
                    tsa_fusion_params.append(v)
                else:
                    normal_params.append(v)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': tsa_fusion_params,
                    'lr': train_opt['optim_g']['lr']
                },
            ]
        else:
            optim_params = []
            for k, v in self.net_g.named_parameters():
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def fix_parames(self, step):
        # fix the weights of normal module
        if self.opt['train'][
                'tsa_iter'] and step <= self.opt['train']['tsa_iter']:
            self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        self.fix_parames(step)

        super(VideoBaseModel, self).optimize_parameters(step)
