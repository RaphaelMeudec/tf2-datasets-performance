import tensorflow as tf

from loaders.utils import select_patch


class IndependantDataLoaderGroupedImageLoading:
    def load_image(self, image_path, dtype=tf.float32):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, dtype)
        image = (image - 0.5) / 2

        return image

    def image_dataset(self, images_paths, dtype=tf.float32):
        dataset = tf.data.Dataset.from_tensor_slices(images_paths)
        dataset = dataset.map(
            lambda x: self.load_image(x, dtype),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        return dataset

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

        blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]

        sharp_dataset = self.image_dataset(sharp_images_path, dtype)
        blur_dataset = self.image_dataset(blur_images_path, dtype)

        dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
        dataset = dataset.map(
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=50)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
