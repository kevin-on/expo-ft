"""Image selection and FOV correction; no device access."""
from copy import deepcopy
import numpy as np
from client.envs.utils import process_image_for_obs


def selected_views(camera_ids):
    result = {}
    for camera_id in camera_ids:
        if not camera_id:
            continue
        serial, eye = camera_id.rsplit('_', 1)
        if eye not in ('left', 'right'):
            raise ValueError(f'Invalid camera eye: {camera_id}')
        result.setdefault(serial, [])
        if eye not in result[serial]:
            result[serial].append(eye)
    return result


def crop_for(camera_id, image, crops, resolution):
    box = crops.get(camera_id, {}).get(resolution)
    if box is None:
        return image, (0, 0)
    if len(box) != 4 or any(type(n) is not int for n in box):
        raise ValueError('Camera crop must be integer [x, y, width, height]')
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > image.shape[1] or y+h > image.shape[0]:
        raise ValueError(f'Crop outside {camera_id} image: {box}')
    return image[y:y+h, x:x+w], (x, y)


def prepare_images(raw, camera_ids, crops, resolutions, image_size):
    """Return owned model images and their adjusted (physical-frame) calibration."""
    images, intrinsics = {}, deepcopy(raw.get('camera_intrinsics', {}))
    for camera_id in camera_ids:
        if not camera_id:
            continue
        frame, (x, y) = crop_for(camera_id, raw['image'][camera_id], crops,
                                resolutions.get(camera_id, '1080p'))
        images[camera_id] = process_image_for_obs(frame, bgr_to_rgb=True, image_size=image_size)
        if camera_id in intrinsics:
            K = np.array(intrinsics[camera_id], dtype=np.float64, copy=True)
            K[0, 2] -= x; K[1, 2] -= y
            if image_size is not None:
                K[0, :] *= image_size[1]/frame.shape[1]
                K[1, :] *= image_size[0]/frame.shape[0]
            intrinsics[camera_id] = K
    return images, intrinsics
