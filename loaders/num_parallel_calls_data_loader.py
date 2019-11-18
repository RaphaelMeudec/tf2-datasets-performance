import tensorflow as tf

from loaders.utils import select_patch


class NumParallelCallsLoader:
    def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
        images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]

        dataset = tf.data.Dataset.from_tensor_slices(images_path)
        dataset = (
            dataset.map(
                lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur")),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(  # Read both sharp and blur files
                lambda sharp_path, blur_path: (
                    tf.io.read_file(sharp_path),
                    tf.io.read_file(blur_path),
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(  # Decode as png both sharp and blur files
                lambda sharp_file, blur_file: (
                    tf.image.decode_png(sharp_file, channels=3),
                    tf.image.decode_png(blur_file, channels=3),
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )
        dataset = (
            dataset.map(  # Convert to float32 both sharp and blur files
                lambda sharp_image, blur_image: (
                    tf.image.convert_image_dtype(sharp_image, tf.float32),
                    tf.image.convert_image_dtype(blur_image, tf.float32),
                )
            )
            .map(  # Load images between [-1, 1] instead of [0, 1]
                lambda sharp_image, blur_image: (
                    (sharp_image - 0.5) * 2,
                    (blur_image - 0.5) * 2,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(  # Select subset of the image
                lambda sharp_image, blur_image: select_patch(
                    sharp_image, blur_image, patch_size[0], patch_size[1]
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()

        return dataset
