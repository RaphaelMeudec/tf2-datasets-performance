import random

import tensorflow as tf

from loaders.utils import select_patch


class BasicPythonGeneratorWithTFOperators:
    def load(
        self,
        dataset_path,
        batch_size=4,
        patch_size=(256, 256),
        dtype=tf.float32,
        n_images=None,
    ):
        sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
        if n_images is not None:
            sharp_images_path = sharp_images_path[0:n_images]

        def load_image(image_path):
            image_file = tf.io.read_file(image_path)
            image = tf.image.decode_png(image_file)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = (image - 0.5) * 2

            return image

        batch_paths = random.choices(sharp_images_path, k=batch_size)
        sharp_images = [load_image(path) for path in batch_paths]
        blur_images = [
            load_image(path.replace("sharp", "blur")) for path in batch_paths
        ]

        while True:
            patches = [
                list(select_patch(sharp, blur, patch_size[0], patch_size[1]))
                for sharp, blur in zip(sharp_images, blur_images)
            ]

            yield tf.convert_to_tensor(
                [patch[0] for patch in patches], dtype
            ), tf.convert_to_tensor([patch[1] for patch in patches], dtype)
