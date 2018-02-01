import numpy as np
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings('ignore', r'Possibly corrupt EXIF data',
                        UserWarning, "PIL.TiffImagePlugin",0)
warnings.filterwarnings('ignore', r'Corrupt EXIF data',
                        UserWarning, "PIL.TiffImagePlugin",0)
from keras import backend as K
import random
from torchvision import transforms
from cv2 import flip
from keras.applications.imagenet_utils import preprocess_input


IMAGE_SIZE = 256
CROP_SIZE = 224

# http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
# http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = flip(img, 0)  # np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img


def randomHorizontalFlip(img, u=0.5):
    shape = img.shape
    if random.random() < u:
        img = flip(img, 1)  # np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img

def randomCrop(img, crop_size=224):
    h, w, c = img.shape
    dy = random.randint(0, h - crop_size)
    dx = random.randint(0, w - crop_size)
    img = img[dy:dy + crop_size, dx:dx + crop_size]
    return img

def cropCenter(img, height, width):
    h, w, c = img.shape
    dx = (h - height) // 2
    dy = (w - width) // 2
    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]
    return img

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def _resize_and_pad(img, desired_size):
    """ Resize this into an image of size (desired_size, desired_size),
        preserving aspect ratio and in the case of a non-square image,
        centering it and padding the smaller side.

    :param img: PIL.Image
    :param desired_size: width and height of the returned image in pixels
    :return: downscaled image
    """

    img.thumbnail((desired_size, desired_size), Image.LANCZOS)
    delta_w = desired_size - img.width
    delta_h = desired_size - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2),
               delta_h - (delta_h // 2))
    new_im = ImageOps.expand(img, padding)

    return new_im

def resize_and_pad_file(filename, desired_size=IMAGE_SIZE):
    """ Resize this into an image of size (desired_size, desired_size),
        preserving aspect ratio and in the case of a non-square image,
        centering it and padding the smaller side.

    :param filename: image filename
    :param desired_size: width and height of the returned image in pixels
    :return: downscaled image
    """
    img = Image.open(filename)
    img = _resize_and_pad(img, desired_size)
    return img

def resize_and_pad(image_arr, desired_size=IMAGE_SIZE):
    """ Resize this into an image of size (desired_size, desired_size),
        preserving aspect ratio and in the case of a non-square image,
        centering it and padding the smaller side.

    :param filename: image filename
    :param desired_size: width and height of the returned image in pixels
    :return: downscaled image
    """
    img = Image.fromarray(image_arr)
    img = _resize_and_pad(img, desired_size)
    return img

def randomCropFlips(size=224):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: randomHorizontalFlip(x, u=0.5)),
        transforms.Lambda(lambda x: randomCrop(x, size)),
    ])
    return transform

def centerCrop(size=224):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cropCenter(x, height=size, width=size)),
    ])
    return transform
