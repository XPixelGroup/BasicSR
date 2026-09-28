import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_img_util():
    if "torchvision.utils" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        utils = types.ModuleType("torchvision.utils")
        utils.make_grid = lambda *args, **kwargs: None
        torchvision.utils = utils
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.utils"] = utils
    path = Path(__file__).resolve().parents[1] / "basicsr" / "utils" / "img_util.py"
    spec = importlib.util.spec_from_file_location("basicsr_img_util", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


img_util = _load_img_util()


class TestTensor2ImgFast(unittest.TestCase):
    def test_a_half_step_is_rounded(self):
        pixel = torch.tensor([0.5, 0.0, 1.0]).view(1, 3, 1, 1)
        image = img_util.tensor2img_fast(pixel, rgb2bgr=False)
        self.assertEqual(int(image[0, 0, 0]), 128)
        self.assertEqual(int(image[0, 0, 1]), 0)
        self.assertEqual(int(image[0, 0, 2]), 255)


if __name__ == "__main__":
    unittest.main()
