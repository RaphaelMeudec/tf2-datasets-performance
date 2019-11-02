import time
import types
from functools import partial
from pathlib import Path

import tensorflow as tf

from loaders import BasicTFDataLoader, BasicPythonGeneratorWithTFOperators
from losses import perceptual_loss
from model import FPNInception

N_ITERATIONS = 20
BATCH_SIZE = 16
PATCH_SIZE = (256, 256)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} tooks {end_time - start_time} seconds")
        return result

    return wrapper


@timeit
def time_dataset(model, dataset, dataset_name, n_iterations):
    training_parameters = {
        "steps_per_epoch": n_iterations,
        "epochs": 1,
        "callbacks": [
            tf.keras.callbacks.TensorBoard(
                log_dir=f"./logs/{dataset_name}", profile_batch=3
            )
        ],
    }

    if isinstance(dataset, types.GeneratorType):
        model.fit_generator(dataset, **training_parameters)
    else:
        model.fit(dataset, **training_parameters)


#
# class BasicTFDataLoader:
#     def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
#         images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
#
#         dataset = tf.data.Dataset.from_tensor_slices(images_path)
#         dataset = (
#             dataset.map(
#                 lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur"))
#             )
#             .map(  # Read both sharp and blur files
#                 lambda sharp_path, blur_path: (
#                     tf.io.read_file(sharp_path),
#                     tf.io.read_file(blur_path),
#                 )
#             )
#             .map(  # Decode as png both sharp and blur files
#                 lambda sharp_file, blur_file: (
#                     tf.image.decode_png(sharp_file, channels=3),
#                     tf.image.decode_png(blur_file, channels=3),
#                 )
#             )
#         )
#         dataset = (
#             dataset.map(  # Convert to float32 both sharp and blur files
#                 lambda sharp_image, blur_image: (
#                     tf.image.convert_image_dtype(sharp_image, tf.float32),
#                     tf.image.convert_image_dtype(blur_image, tf.float32),
#                 )
#             )
#             .map(  # Load images between [-1, 1] instead of [0, 1]
#                 lambda sharp_image, blur_image: (
#                     (sharp_image - 0.5) * 2,
#                     (blur_image - 0.5) * 2,
#                 )
#             )
#             .map(  # Select subset of the image
#                 lambda sharp_image, blur_image: select_patch(
#                     sharp_image, blur_image, patch_size[0], patch_size[1]
#                 )
#             )
#         )
#
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat()
#
#         return dataset
#
#
# class NumParallelCallsLoader:
#     def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
#         images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
#
#         dataset = tf.data.Dataset.from_tensor_slices(images_path)
#         dataset = (
#             dataset.map(
#                 lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur")),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Read both sharp and blur files
#                 lambda sharp_path, blur_path: (
#                     tf.io.read_file(sharp_path),
#                     tf.io.read_file(blur_path),
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Decode as png both sharp and blur files
#                 lambda sharp_file, blur_file: (
#                     tf.image.decode_png(sharp_file, channels=3),
#                     tf.image.decode_png(blur_file, channels=3),
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#         dataset = (
#             dataset.map(  # Convert to float32 both sharp and blur files
#                 lambda sharp_image, blur_image: (
#                     tf.image.convert_image_dtype(sharp_image, tf.float32),
#                     tf.image.convert_image_dtype(blur_image, tf.float32),
#                 )
#             )
#             .map(  # Load images between [-1, 1] instead of [0, 1]
#                 lambda sharp_image, blur_image: (
#                     (sharp_image - 0.5) * 2,
#                     (blur_image - 0.5) * 2,
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Select subset of the image
#                 lambda sharp_image, blur_image: select_patch(
#                     sharp_image, blur_image, patch_size[0], patch_size[1]
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat()
#
#         return dataset
#
#
# class PrefetchLoader:
#     def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
#         images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
#
#         dataset = tf.data.Dataset.from_tensor_slices(images_path)
#         dataset = (
#             dataset.map(
#                 lambda path: (path, tf.strings.regex_replace(path, "sharp", "blur")),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Read both sharp and blur files
#                 lambda sharp_path, blur_path: (
#                     tf.io.read_file(sharp_path),
#                     tf.io.read_file(blur_path),
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Decode as png both sharp and blur files
#                 lambda sharp_file, blur_file: (
#                     tf.image.decode_png(sharp_file, channels=3),
#                     tf.image.decode_png(blur_file, channels=3),
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#         dataset = (
#             dataset.map(  # Convert to float32 both sharp and blur files
#                 lambda sharp_image, blur_image: (
#                     tf.image.convert_image_dtype(sharp_image, tf.float32),
#                     tf.image.convert_image_dtype(blur_image, tf.float32),
#                 )
#             )
#             .map(  # Load images between [-1, 1] instead of [0, 1]
#                 lambda sharp_image, blur_image: (
#                     (sharp_image - 0.5) * 2,
#                     (blur_image - 0.5) * 2,
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(  # Select subset of the image
#                 lambda sharp_image, blur_image: select_patch(
#                     sharp_image, blur_image, patch_size[0], patch_size[1]
#                 ),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat()
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#         return dataset
#
#
# class TwoDatasetLoader:
#     def image_dataset(self, images_paths):
#         dataset = tf.data.Dataset.from_tensor_slices(images_paths)
#         dataset = (
#             dataset.map(
#                 tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
#             )
#             .map(tf.image.decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#             .map(
#                 lambda x: tf.image.convert_image_dtype(x, tf.float32),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(
#                 lambda x: (x - 0.5) * 2,
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#
#         return dataset
#
#     def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
#         sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
#         blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]
#
#         sharp_dataset = self.image_dataset(sharp_images_path)
#         blur_dataset = self.image_dataset(blur_images_path)
#
#         dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
#         dataset = dataset.map(
#             lambda sharp_image, blur_image: select_patch(
#                 sharp_image, blur_image, patch_size[0], patch_size[1]
#             ),
#             num_parallel_calls=tf.data.experimental.AUTOTUNE,
#         )
#
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat()
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#         return dataset
#
#
# class TwoDatasetLoader:
#     def image_dataset(self, images_paths):
#         dataset = tf.data.Dataset.from_tensor_slices(images_paths)
#         dataset = (
#             dataset.map(
#                 tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
#             )
#             .map(tf.image.decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#             .map(
#                 lambda x: tf.image.convert_image_dtype(x, tf.float32),
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#             .map(
#                 lambda x: (x - 0.5) * 2,
#                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
#             )
#         )
#
#         return dataset
#
#     def load(self, dataset_path, batch_size=4, patch_size=(256, 256)):
#         sharp_images_path = [str(path) for path in dataset_path.glob("*/sharp/*.png")]
#         blur_images_path = [path.replace("sharp", "blur") for path in sharp_images_path]
#
#         sharp_dataset = self.image_dataset(sharp_images_path).cache()
#         blur_dataset = self.image_dataset(blur_images_path).cache()
#
#         dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
#         dataset = dataset.map(
#             lambda sharp_image, blur_image: select_patch(
#                 sharp_image, blur_image, patch_size[0], patch_size[1]
#             ),
#             num_parallel_calls=tf.data.experimental.AUTOTUNE,
#         )
#
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.repeat()
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#         return dataset


if __name__ == "__main__":
    vgg = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights="imagenet", input_shape=(*PATCH_SIZE, 3)
    )
    loss_model = tf.keras.models.Model(
        inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
    )
    loss = partial(perceptual_loss, loss_model=loss_model)

    model = FPNInception(num_filters=128, num_filters_fpn=256)
    model.compile(optimizer="adam", loss=loss)

    dataset_path = Path("/home/raph/Projects/tf2-deblurgan-v2/datasets/gopro/train")
    print("Python generator")
    time_dataset(
        model=model,
        dataset=BasicPythonGeneratorWithTFOperators().load(
            dataset_path, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE
        ),
        n_iterations=N_ITERATIONS,
    )
    print("Initial Loader")
    time_dataset(
        model=model,
        dataset=BasicTFDataLoader().load(
            dataset_path, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE
        ),
        n_iterations=N_ITERATIONS,
    )
    # print("NumParallelCalls Loader")
    # time_dataset(NumParallelCallsLoader().load(dataset_path), N_ITERATIONS)
    # print("Prefetch Loader")
    # time_dataset(PrefetchLoader().load(dataset_path), N_ITERATIONS)
    # print("TwoDataset Loader")
    # time_dataset(TwoDatasetLoader().load(dataset_path), N_ITERATIONS)
