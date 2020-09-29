from basicsr.models.srgan_model import SRGANModel
from basicsr.models.video_base_model import VideoBaseModel


class VideoGANModel(SRGANModel, VideoBaseModel):
    """Video GAN model.

    Use multiple inheritance.
    It will first use the functions of SRGANModel:
        init_training_settings
        setup_optimizers
        optimize_parameters
        save
    Then find functions in VideoBaseModel.
    """
