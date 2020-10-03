import argparse
import cv2
import glob
import mmcv
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from skimage import io
from skimage import transform as trans

from basicsr.models.archs.dfdnet_arch import DFDNet
from basicsr.utils import tensor2img

try:
    import dlib
except ImportError:
    print('Please install dlib before testing face restoration.'
          'Reference:　https://github.com/davisking/dlib')


class FaceRestorationHelper(object):
    """Helper for the face restoration pipeline."""

    def __init__(self, upscale_factor, face_template_path, out_size=512):
        self.upscale_factor = upscale_factor
        self.out_size = (out_size, out_size)

        # standard 5 landmarks for FFHQ faces with 1024 x 1024
        self.face_template = np.load(face_template_path) / (1024 // out_size)
        # for estimation the 2D similarity transformation
        self.similarity_trans = trans.SimilarityTransform()

        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []

    def init_dlib(self, detection_path, landmark5_path, landmark68_path):
        """Initialize the dlib detectors and predictors."""
        self.face_detector = dlib.cnn_face_detection_model_v1(detection_path)
        self.shape_predictor_5 = dlib.shape_predictor(landmark5_path)
        self.shape_predictor_68 = dlib.shape_predictor(landmark68_path)

    def free_dlib_gpu_memory(self):
        del self.face_detector
        del self.shape_predictor_5
        del self.shape_predictor_68

    def read_input_image(self, img_path):
        # self.input_img is Numpy array, (h, w, c) with RGB order
        self.input_img = dlib.load_rgb_image(img_path)

    def detect_faces(self, img_path, upsample_num_times=1):
        """
        Args:
            img_path (str): Image path.
            upsample_num_times (int): Upsamples the image before running the
                face detector

        Returns:
            int: Number of detected faces.
        """
        self.read_input_image(img_path)
        self.det_faces = self.face_detector(self.input_img, upsample_num_times)
        if len(self.det_faces) == 0:
            print('No face detected. Try to increase upsample_num_times.')
        return len(self.det_faces)

    def get_face_landmarks_5(self):
        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)
        return len(self.all_landmarks_5)

    def get_face_landmarks_68(self):
        """Get 68 densemarks for cropped images.

        Should only have one face at most in the cropped image.
        """
        num_detected_face = 0
        for idx, face in enumerate(self.cropped_faces):
            # face detection
            det_face = self.face_detector(face, 1)  # TODO: can we remove it
            if len(det_face) == 0:
                print(f'Cannot find faces in cropped image with index {idx}.')
                self.all_landmarks_68.append(None)
            elif len(det_face) == 1:
                shape = self.shape_predictor_68(face, det_face[0].rect)
                landmark = np.array([[part.x, part.y]
                                     for part in shape.parts()])
                self.all_landmarks_68.append(landmark)
                num_detected_face += 1
            else:
                print('Should only have one face at most.')
        return num_detected_face

    def warp_crop_faces(self, save_cropped_path=None):
        """Get affine matrix, warp and cropped faces.

        Also get inverse affine matrix for post-processing.
        """
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            self.similarity_trans.estimate(landmark, self.face_template)
            affine_matrix = self.similarity_trans.params[0:2, :]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            cropped_face = cv2.warpAffine(self.input_img, affine_matrix,
                                          self.out_size)
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path, ext = os.path.splitext(save_cropped_path)
                save_path = f'{path}_{idx:02d}{ext}'
                mmcv.imwrite(mmcv.rgb2bgr(cropped_face), save_path)

            # get inverse affine matrix
            self.similarity_trans.estimate(self.face_template,
                                           landmark * self.upscale_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(inverse_affine)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path):
        # operate in the BGR order
        input_img = mmcv.rgb2bgr(self.input_img)
        h, w, _ = input_img.shape
        h_up, w_up = h * self.upscale_factor, w * self.upscale_factor
        # simply resize the background
        upsample_img = cv2.resize(input_img, (w_up, h_up))
        for restored_face, inverse_affine in zip(self.restored_faces,
                                                 self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine,
                                          (w_up, h_up))
            mask = np.ones((*self.out_size, 3), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask,
                np.ones((2 * self.upscale_factor, 2 * self.upscale_factor),
                        np.uint8))
            inv_restored_remove_border = inv_mask_erosion * inv_restored
            total_face_area = np.sum(inv_mask_erosion) // 3
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(
                inv_mask_erosion,
                np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center,
                                             (blur_size + 1, blur_size + 1), 0)
            upsample_img = inv_soft_mask * inv_restored_remove_border + (
                1 - inv_soft_mask) * upsample_img
        mmcv.imwrite(upsample_img.astype(np.uint8), save_path)

    def clean_all(self):
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []


def get_part_location(landmarks):
    """Get part locations from landmarks."""
    map_left_eye = list(np.hstack((range(17, 22), range(36, 42))))
    map_right_eye = list(np.hstack((range(22, 27), range(42, 48))))
    map_nose = list(range(29, 36))
    map_mouth = list(range(48, 68))

    # left eye
    mean_left_eye = np.mean(landmarks[map_left_eye], 0)  # (x, y)
    half_len_left_eye = np.max((np.max(
        np.max(landmarks[map_left_eye], 0) -
        np.min(landmarks[map_left_eye], 0)) / 2, 16))  # A number
    loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1,
                              mean_left_eye + half_len_left_eye)).astype(int)
    loc_left_eye = torch.from_numpy(loc_left_eye).unsqueeze(0)
    # (1, 4), the four numbers forms two  coordinates in the diagonal

    # right eye
    mean_right_eye = np.mean(landmarks[map_right_eye], 0)
    half_len_right_eye = np.max((np.max(
        np.max(landmarks[map_right_eye], 0) -
        np.min(landmarks[map_right_eye], 0)) / 2, 16))
    loc_right_eye = np.hstack(
        (mean_right_eye - half_len_right_eye + 1,
         mean_right_eye + half_len_right_eye)).astype(int)
    loc_right_eye = torch.from_numpy(loc_right_eye).unsqueeze(0)
    # nose
    mean_nose = np.mean(landmarks[map_nose], 0)
    half_len_nose = np.max((np.max(
        np.max(landmarks[map_nose], 0) - np.min(landmarks[map_nose], 0)) / 2,
                            16))  # noqa: E126
    loc_nose = np.hstack(
        (mean_nose - half_len_nose + 1, mean_nose + half_len_nose)).astype(int)
    loc_nose = torch.from_numpy(loc_nose).unsqueeze(0)
    # mouth
    mean_mouth = np.mean(landmarks[map_mouth], 0)
    half_len_mouth = np.max((np.max(
        np.max(landmarks[map_mouth], 0) - np.min(landmarks[map_mouth], 0)) / 2,
                             16))  # noqa: E126
    loc_mouth = np.hstack((mean_mouth - half_len_mouth + 1,
                           mean_mouth + half_len_mouth)).astype(int)
    loc_mouth = torch.from_numpy(loc_mouth).unsqueeze(0)

    return loc_left_eye, loc_right_eye, loc_nose, loc_mouth


if __name__ == '__main__':
    """We try to align to the official codes. But there are still slight
    differences: 1) we use dlib for 68 landmark detection; 2) the used image
    package are different (especially for reading and writing.)
    """
    device = 'cuda'
    parser = argparse.ArgumentParser()

    parser.add_argument('--upscale_factor', type=int, default=2)
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/DFDNet/DFDNet_official-d1fa5650.pth')
    parser.add_argument(
        '--dict_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/DFDNet/DFDNet_dict_512-f79685f0.pth')
    parser.add_argument('--test_path', type=str, default='datasets/TestWhole')
    parser.add_argument('--upsample_num_times', type=int, default=1)
    # The official codes use skimage.io to read the cropped images from disk
    # instead of directly using the intermediate results in the memory (as we
    # do). Such a different operation brings slight differences due to
    # skimage.io. For aligning with the official results, we could set the
    # official_adaption to True.
    parser.add_argument('--official_adaption', type=bool, default=True)

    # The following are the paths for face template and dlib models
    parser.add_argument(
        '--face_template_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/DFDNet/FFHQ_5_landmarks_template_1024-90a00515.npy'  # noqa: E501
    )
    parser.add_argument(
        '--detection_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/dlib/mmod_human_face_detector-4cb19393.dat'  # noqa: E501
    )
    parser.add_argument(
        '--landmark5_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/dlib/shape_predictor_5_face_landmarks-c4b1e980.dat'  # noqa: E501
    )
    parser.add_argument(
        '--landmark68_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat'  # noqa: E501
    )

    args = parser.parse_args()
    result_root = f'results/DFDNet/{args.test_path.split("/")[-1]}'

    # set up the DFDNet
    net = DFDNet(64, dict_path=args.dict_path).to(device)
    checkpoint = torch.load(
        args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    net.eval()

    save_crop_root = os.path.join(result_root, 'cropped_faces')
    save_restore_root = os.path.join(result_root, 'restored_faces')
    save_final_root = os.path.join(result_root, 'final_results')

    face_helper = FaceRestorationHelper(
        args.upscale_factor, args.face_template_path, out_size=512)

    # scan all the jpg and png images
    for img_path in glob.glob(os.path.join(args.test_path, '*.[jp][pn]g')):
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} image ...')
        save_crop_path = os.path.join(save_crop_root, img_name)

        face_helper.init_dlib(args.detection_path, args.landmark5_path,
                              args.landmark68_path)
        # detect faces
        num_det_faces = face_helper.detect_faces(
            img_path, upsample_num_times=args.upsample_num_times)
        # get 5 face landmarks for each face
        num_landmarks = face_helper.get_face_landmarks_5()
        print(f'\tDetect {num_det_faces} faces, {num_landmarks} landmarks.')
        # warp and crop each face
        face_helper.warp_crop_faces(save_crop_path)

        if args.official_adaption:
            path, ext = os.path.splitext(save_crop_path)
            pathes = sorted(glob.glob(f'{path}_[0-9]*{ext}'))
            cropped_faces = [io.imread(path) for path in pathes]
        else:
            cropped_faces = face_helper.cropped_faces

        # get 68 landmarks for each cropped face
        num_landmarks = face_helper.get_face_landmarks_68()
        print(f'\tDetect {num_landmarks} faces for 68 landmarks.')

        face_helper.free_dlib_gpu_memory()

        print('\tFace restoration ...')
        # face restoration for each cropped face
        for idx, (cropped_face, landmarks) in enumerate(
                zip(cropped_faces, face_helper.all_landmarks_68)):
            if landmarks is None:
                print(f'Landmarks is None, skip cropped faces with idx {idx}.')
            else:
                # prepare data
                part_locations = get_part_location(landmarks)
                cropped_face = transforms.ToTensor()(cropped_face)
                cropped_face = transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))(
                                                        cropped_face)
                cropped_face = cropped_face.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = net(cropped_face, part_locations)
                    im = tensor2img(output, min_max=(-1, 1))
                    del output
                torch.cuda.empty_cache()
                path, ext = os.path.splitext(
                    os.path.join(save_restore_root, img_name))
                save_path = f'{path}_{idx:02d}{ext}'
                mmcv.imwrite(im, save_path)
                face_helper.add_restored_face(im)

        print('\tGenerate the final result ...')
        # paste each restored face to the input image
        face_helper.paste_faces_to_input_image(
            os.path.join(save_final_root, img_name))

        # clean all the intermediate results to process the next image
        face_helper.clean_all()

    print(f'\nAll results are saved in {result_root}')
