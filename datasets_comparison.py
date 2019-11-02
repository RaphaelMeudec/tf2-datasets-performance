import time
import types
from functools import partial
from pathlib import Path

import tensorflow as tf

from loaders import (
    BasicTFDataLoader,
    BasicPythonGeneratorWithTFOperators,
    NumParallelCallsLoader,
    PrefetchLoader,
    IndependantDataLoader,
)
from losses import perceptual_loss
from model import FPNInception

N_EPOCHS = 1
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
        "epochs": N_EPOCHS,
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
        dataset_name="python_generator",
        n_iterations=N_ITERATIONS,
    )
    print("Initial Loader")
    time_dataset(
        model=model,
        dataset=BasicTFDataLoader().load(
            dataset_path, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE
        ),
        dataset_name="tf2_basic_loader",
        n_iterations=N_ITERATIONS,
    )
    print("NumParallelCalls Loader")
    time_dataset(
        model=model,
        dataset=NumParallelCallsLoader().load(dataset_path),
        dataset_name="num_parallel_calls",
        n_iterations=N_ITERATIONS,
    )
    print("Prefetch Loader")
    time_dataset(
        model=model,
        dataset=PrefetchLoader().load(dataset_path),
        dataset_name="prefetch",
        n_iterations=N_ITERATIONS,
    )
    print("Independant Loader")
    time_dataset(
        model=model,
        dataset=IndependantDataLoader().load(dataset_path),
        dataset_name="independant_loaders",
        n_iterations=N_ITERATIONS,
    )
