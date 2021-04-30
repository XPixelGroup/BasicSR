import argparse
import glob
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from skimage import io

from basicsr.archs.dfdnet_arch import DFDNet
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.face_util import FaceRestorationHelper


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    parser.add_argument('--save_inverse_affine', action='store_true')
    parser.add_argument('--only_keep_largest', action='store_true')
    # The official codes use skimage.io to read the cropped images from disk
    # instead of directly using the intermediate results in the memory (as we
    # do). Such a different operation brings slight differences due to
    # skimage.io. For aligning with the official results, we could set the
    # official_adaption to True.
    parser.add_argument('--official_adaption', type=bool, default=True)

    # The following are the paths for dlib models
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
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    result_root = f'results/DFDNet/{os.path.basename(args.test_path)}'

    # set up the DFDNet
    net = DFDNet(64, dict_path=args.dict_path).to(device)
    checkpoint = torch.load(
        args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['params'])
    net.eval()

    save_crop_root = os.path.join(result_root, 'cropped_faces')
    save_inverse_affine_root = os.path.join(result_root, 'inverse_affine')
    os.makedirs(save_inverse_affine_root, exist_ok=True)
    save_restore_root = os.path.join(result_root, 'restored_faces')
    save_final_root = os.path.join(result_root, 'final_results')

    face_helper = FaceRestorationHelper(args.upscale_factor, face_size=512)

    # scan all the jpg and png images
    for img_path in sorted(
            glob.glob(os.path.join(args.test_path, '*.[jp][pn]g'))):
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} image ...')
        save_crop_path = os.path.join(save_crop_root, img_name)
        if args.save_inverse_affine:
            save_inverse_affine_path = os.path.join(save_inverse_affine_root,
                                                    img_name)
        else:
            save_inverse_affine_path = None

        face_helper.init_dlib(args.detection_path, args.landmark5_path,
                              args.landmark68_path)
        # detect faces
        num_det_faces = face_helper.detect_faces(
            img_path,
            upsample_num_times=args.upsample_num_times,
            only_keep_largest=args.only_keep_largest)
        # get 5 face landmarks for each face
        num_landmarks = face_helper.get_face_landmarks_5()
        print(f'\tDetect {num_det_faces} faces, {num_landmarks} landmarks.')
        # warp and crop each face
        face_helper.warp_crop_faces(save_crop_path, save_inverse_affine_path)

        if args.official_adaption:
            path, ext = os.path.splitext(save_crop_path)
            pathes = sorted(glob.glob(f'{path}_[0-9]*.png'))
            cropped_faces = [io.imread(path) for path in pathes]
        else:
            cropped_faces = face_helper.cropped_faces

        # get 68 landmarks for each cropped face
        num_landmarks = face_helper.get_face_landmarks_68()
        print(f'\tDetect {num_landmarks} faces for 68 landmarks.')

        face_helper.free_dlib_gpu_memory()

        print('\tFace restoration ...')
        # face restoration for each cropped face
        assert len(cropped_faces) == len(face_helper.all_landmarks_68)
        for idx, (cropped_face, landmarks) in enumerate(
                zip(cropped_faces, face_helper.all_landmarks_68)):
            if landmarks is None:
                print(f'Landmarks is None, skip cropped faces with idx {idx}.')
                # just copy the cropped faces to the restored faces
                restored_face = cropped_face
            else:
                # prepare data
                part_locations = get_part_location(landmarks)
                cropped_face = transforms.ToTensor()(cropped_face)
                cropped_face = transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))(
                                                        cropped_face)
                cropped_face = cropped_face.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face, part_locations)
                        restored_face = tensor2img(output, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f'DFDNet inference fail: {e}')
                    restored_face = tensor2img(cropped_face, min_max=(-1, 1))

            path = os.path.splitext(os.path.join(save_restore_root,
                                                 img_name))[0]
            save_path = f'{path}_{idx:02d}.png'
            imwrite(restored_face, save_path)
            face_helper.add_restored_face(restored_face)

        print('\tGenerate the final result ...')
        # paste each restored face to the input image
        face_helper.paste_faces_to_input_image(
            os.path.join(save_final_root, img_name))

        # clean all the intermediate results to process the next image
        face_helper.clean_all()

    print(f'\nAll results are saved in {result_root}')
