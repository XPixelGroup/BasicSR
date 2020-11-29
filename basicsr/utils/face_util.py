import cv2
import numpy as np
import os
import torch
from skimage import transform as trans

from basicsr.utils import imwrite

try:
    import dlib
except ImportError:
    print('Please install dlib before testing face restoration.'
          'Reference:　https://github.com/davisking/dlib')


class FaceRestorationHelper(object):
    """Helper for the face restoration pipeline."""

    def __init__(self, upscale_factor, face_size=512):
        self.upscale_factor = upscale_factor
        self.face_size = (face_size, face_size)

        # standard 5 landmarks for FFHQ faces with 1024 x 1024
        self.face_template = np.array([[686.77227723, 488.62376238],
                                       [586.77227723, 493.59405941],
                                       [337.91089109, 488.38613861],
                                       [437.95049505, 493.51485149],
                                       [513.58415842, 678.5049505]])
        self.face_template = self.face_template / (1024 // face_size)
        # for estimation the 2D similarity transformation
        self.similarity_trans = trans.SimilarityTransform()

        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.save_png = True

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

    def detect_faces(self,
                     img_path,
                     upsample_num_times=1,
                     only_keep_largest=False):
        """
        Args:
            img_path (str): Image path.
            upsample_num_times (int): Upsamples the image before running the
                face detector

        Returns:
            int: Number of detected faces.
        """
        self.read_input_image(img_path)
        det_faces = self.face_detector(self.input_img, upsample_num_times)
        if len(det_faces) == 0:
            print('No face detected. Try to increase upsample_num_times.')
        else:
            if only_keep_largest:
                print('Detect several faces and only keep the largest.')
                face_areas = []
                for i in range(len(det_faces)):
                    face_area = (det_faces[i].rect.right() -
                                 det_faces[i].rect.left()) * (
                                     det_faces[i].rect.bottom() -
                                     det_faces[i].rect.top())
                    face_areas.append(face_area)
                largest_idx = face_areas.index(max(face_areas))
                self.det_faces = [det_faces[largest_idx]]
            else:
                self.det_faces = det_faces
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
            det_face = self.face_detector(face, 1)  # TODO: can we remove it?
            if len(det_face) == 0:
                print(f'Cannot find faces in cropped image with index {idx}.')
                self.all_landmarks_68.append(None)
            else:
                if len(det_face) > 1:
                    print('Detect several faces in the cropped face. Use the '
                          ' largest one. Note that it will also cause overlap '
                          'during paste_faces_to_input_image.')
                    face_areas = []
                    for i in range(len(det_face)):
                        face_area = (det_face[i].rect.right() -
                                     det_face[i].rect.left()) * (
                                         det_face[i].rect.bottom() -
                                         det_face[i].rect.top())
                        face_areas.append(face_area)
                    largest_idx = face_areas.index(max(face_areas))
                    face_rect = det_face[largest_idx].rect
                else:
                    face_rect = det_face[0].rect
                shape = self.shape_predictor_68(face, face_rect)
                landmark = np.array([[part.x, part.y]
                                     for part in shape.parts()])
                self.all_landmarks_68.append(landmark)
                num_detected_face += 1

        return num_detected_face

    def warp_crop_faces(self,
                        save_cropped_path=None,
                        save_inverse_affine_path=None):
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
                                          self.face_size)
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path, ext = os.path.splitext(save_cropped_path)
                if self.save_png:
                    save_path = f'{path}_{idx:02d}.png'
                else:
                    save_path = f'{path}_{idx:02d}{ext}'

                imwrite(
                    cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), save_path)

            # get inverse affine matrix
            self.similarity_trans.estimate(self.face_template,
                                           landmark * self.upscale_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path):
        # operate in the BGR order
        input_img = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2BGR)
        h, w, _ = input_img.shape
        h_up, w_up = h * self.upscale_factor, w * self.upscale_factor
        # simply resize the background
        upsample_img = cv2.resize(input_img, (w_up, h_up))
        assert len(self.restored_faces) == len(self.inverse_affine_matrices), (
            'length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces,
                                                 self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine,
                                          (w_up, h_up))
            mask = np.ones((*self.face_size, 3), dtype=np.float32)
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
        if self.save_png:
            save_path = save_path.replace('.jpg',
                                          '.png').replace('.jpeg', '.png')
        imwrite(upsample_img.astype(np.uint8), save_path)

    def clean_all(self):
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
