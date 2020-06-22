import logging

from torch.nn.parallel import DistributedDataParallel

from basicsr.models.video_base_model import VideoBaseModel

logger = logging.getLogger('basicsr')


class EDVRModel(VideoBaseModel):
    """EDVR Model.
    """

    def __init__(self, opt):
        super(EDVRModel, self).__init__(opt)
        self.train_tsa_iter = opt['train']['tsa_iter']

    # def setup_optimizers(self):
    # TODO: set dcn a differnet learning rate.

    def optimize_parameters(self, current_iter):
        if self.train_tsa_iter:
            if current_iter == 1:
                logger.info(
                    f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                logger.info('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True
                if isinstance(self.net_g, DistributedDataParallel):
                    logger.info('Set net_g.find_unused_parameters = False.')
                    self.net_g.find_unused_parameters = False

        super(VideoBaseModel, self).optimize_parameters(current_iter)
