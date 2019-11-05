import tensorflow as tf
from loguru import logger

from loaders.utils import select_patch


class TFRecordDataLoader:
    def image_dataset(self, images_paths):
        dataset = tf.data.Dataset.from_tensor_slices(images_paths)
        dataset = (
            dataset.map(
                tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .map(tf.image.decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(
                lambda x: tf.image.convert_image_dtype(x, tf.float32),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(
                lambda x: (x - 0.5) * 2,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        return dataset

    def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
        sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
        blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]

        sharp_dataset = self.image_dataset(sharp_images_path).cache()
        blur_dataset = self.image_dataset(blur_images_path).cache()

        dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))

        filename = "gopro.tfrecords"
        writer = tf.data.experimental.TFRecordWriter(filename)
        logger.info("Start writing dataset as TFRecords")
        writer.write(dataset)
        logger.info("Done saving dataset.")

        record_dataset = tf.data.TFRecordDataset([filename]).cache()

        record_dataset = record_dataset.map(
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        record_dataset = record_dataset.batch(batch_size)
        record_dataset = record_dataset.repeat()
        record_dataset = record_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return record_dataset
