import cv2
import os
from tqdm import tqdm


class Mosaic16x:
    """
    Mosaic16x: A customized image augmentor for 16-pixel mosaic
    By default it replaces each pixel value with the mean value
    of its 16x16 neighborhood
    """

    def augment_image(self, x):
        h, w = x.shape[:2]
        x = x.astype('float')  # avoid overflow for uint8
        irange, jrange = (h + 15) // 16, (w + 15) // 16
        for i in range(irange):
            for j in range(jrange):
                mean = x[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16].mean(axis=(0, 1))
                x[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = mean

        return x.astype('uint8')


class DegradationSimulator:
    """
    Generating training/testing data pairs on the fly.
    The degradation script is aligned with HiFaceGAN paper settings.

    Args:
        opt(str | op): Config for degradation script, with degradation type and parameters
        Custom degradation is possible by passing an inherited class from ia.augmentors
    """

    def __init__(self, ):
        import imgaug.augmenters as ia
        self.default_deg_templates = {
            'sr4x':
            ia.Sequential([
                # It's almost like a 4x bicubic downsampling
                ia.Resize((0.25000, 0.25001), cv2.INTER_AREA),
                ia.Resize({
                    'height': 512,
                    'width': 512
                }, cv2.INTER_CUBIC),
            ]),
            'sr4x8x':
            ia.Sequential([
                ia.Resize((0.125, 0.25), cv2.INTER_AREA),
                ia.Resize({
                    'height': 512,
                    'width': 512
                }, cv2.INTER_CUBIC),
            ]),
            'denoise':
            ia.OneOf([
                ia.AdditiveGaussianNoise(scale=(20, 40), per_channel=True),
                ia.AdditiveLaplaceNoise(scale=(20, 40), per_channel=True),
                ia.AdditivePoissonNoise(lam=(15, 30), per_channel=True),
            ]),
            'deblur':
            ia.OneOf([
                ia.MotionBlur(k=(10, 20)),
                ia.GaussianBlur((3.0, 8.0)),
            ]),
            'jpeg':
            ia.JpegCompression(compression=(50, 85)),
            '16x':
            Mosaic16x(),
        }

        rand_deg_list = [
            self.default_deg_templates['deblur'],
            self.default_deg_templates['denoise'],
            self.default_deg_templates['jpeg'],
            self.default_deg_templates['sr4x8x'],
        ]
        self.default_deg_templates['face_renov'] = ia.Sequential(rand_deg_list, random_order=True)

    def create_training_dataset(self, deg, gt_folder, lq_folder=None):
        from imgaug.augmenters.meta import Augmenter  # baseclass
        """
        Create a degradation simulator and apply it to GT images on the fly
        Save the degraded result in the lq_folder (if None, name it as GT_deg)
        """
        if not lq_folder:
            suffix = deg if isinstance(deg, str) else 'custom'
            lq_folder = '_'.join([gt_folder.replace('gt', 'lq'), suffix])
        print(lq_folder)
        os.makedirs(lq_folder, exist_ok=True)

        if isinstance(deg, str):
            assert deg in self.default_deg_templates, (
                f'Degration type {deg} not recognized: {"|".join(list(self.default_deg_templates.keys()))}')
            deg = self.default_deg_templates[deg]
        else:
            assert isinstance(deg, Augmenter), f'Deg must be either str|Augmenter, got {deg}'

        names = os.listdir(gt_folder)
        for name in tqdm(names):
            gt = cv2.imread(os.path.join(gt_folder, name))
            lq = deg.augment_image(gt)
            # pack = np.concatenate([lq, gt], axis=0)
            cv2.imwrite(os.path.join(lq_folder, name), lq)

        print('Dataset prepared.')


if __name__ == '__main__':
    simuator = DegradationSimulator()
    gt_folder = 'datasets/FFHQ_512_gt'
    deg = 'sr4x'
    simuator.create_training_dataset(deg, gt_folder)
