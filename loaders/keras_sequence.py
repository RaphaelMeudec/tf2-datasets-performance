import math

import tensorflow as tf

from loaders.utils import select_patch


class KerasSequence(tf.keras.Sequence):
    def __init__(self, dataset_path, batch_size, patch_size, **kwargs):
        super(KerasSequence, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.sharp_images_path = [
            str(path) for path in dataset_path.glob("*/sharp/*.png")
        ]
        self.blur_images_path = [
            str(path) for path in dataset_path.glob("*/sharp/*.png")
        ]
        self.patch_size = patch_size

    def __len__(self):
        return math.ceil(len(self.sharp_images_path) // self.batch_size)

    def __getitem__(self, item):
        first_index = item * self.batch_size
        last_index = (item + 1) * self.batch_size

        def load_image(image_path):
            image_file = tf.io.read_file(image_path)
            image = tf.image.decode_png(image_file)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = (image - 0.5) * 2

            return image

        sharp_paths = self.sharp_images_path[first_index:last_index]
        blur_paths = [path.replace("sharp", "blur") for path in sharp_paths]

        sharp_images = [load_image(path) for path in sharp_paths]
        blur_images = [load_image(path) for path in blur_paths]

        patches = [
            list(select_patch(sharp, blur, self.patch_size[0], self.patch_size[1]))
            for sharp, blur in zip(sharp_images, blur_images)
        ]

        return (
            tf.convert_to_tensor([patch[0] for patch in patches], tf.float32),
            tf.convert_to_tensor([patch[1] for patch in patches], tf.float32),
        )
