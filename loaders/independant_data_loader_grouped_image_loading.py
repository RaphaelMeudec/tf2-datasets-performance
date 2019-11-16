import tensorflow as tf

from loaders.utils import select_patch


class IndependantDataLoaderGroupedImageLoading:
    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (image - 0.5) / 2

        return image

    def image_dataset(self, images_paths):
        dataset = tf.data.Dataset.from_tensor_slices(images_paths)
        dataset = dataset.map(
            self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return dataset

    def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
        sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
        blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]

        sharp_dataset = self.image_dataset(sharp_images_path)
        blur_dataset = self.image_dataset(blur_images_path)

        dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
        dataset = dataset.map(
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
