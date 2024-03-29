import time
import types
from functools import partial
from pathlib import Path

import click
import tensorflow as tf
from loguru import logger

import loaders
from losses import perceptual_loss
from model import FPNInception

PATCH_SIZE = (256, 256)


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info("This took {duration} seconds", duration=duration)
        return result

    return wrapper


@timeit
def time_dataset(model, dataset, dataset_name, steps_per_epoch, epochs):
    training_parameters = {
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "callbacks": [
            tf.keras.callbacks.TensorBoard(
                log_dir=f"./logs/{dataset_name}", profile_batch=3
            )
        ],
        "use_multiprocessing": True,
        "workers": 5,
    }

    if isinstance(dataset, types.GeneratorType):
        model.fit_generator(dataset, **training_parameters)
    else:
        model.fit(dataset, **training_parameters)


@click.command()
@click.option("--epochs", type=int, default=1, help="Number of epochs to perform")
@click.option(
    "--steps_per_epoch", type=int, default=20, help="Number of steps per epoch"
)
@click.option("--batch_size", type=int, default=16, help="Size of mini-batch")
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    default="/home/raph/Projects/tf2-deblurgan-v2/datasets/gopro/train",
    help="Path to gopro train dataset",
)
def run_analysis(epochs, steps_per_epoch, batch_size, dataset_path):
    logger.add(
        f"epochs_{epochs}_steps_{steps_per_epoch}_batch_{batch_size}_keras_sequence.log"
    )

    vgg = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights="imagenet", input_shape=(*PATCH_SIZE, 3)
    )
    loss_model = tf.keras.models.Model(
        inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
    )
    loss = partial(perceptual_loss, loss_model=loss_model)

    model = FPNInception(num_filters=128, num_filters_fpn=256)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    model.compile(optimizer=optimizer, loss=loss)
    # Random first fit to initialize everything
    logger.info("Warm-up training to initialize graph.")

    model.fit(
        tf.random.uniform((1, *PATCH_SIZE, 3), dtype=tf.float32),
        tf.random.uniform((1, *PATCH_SIZE, 3), dtype=tf.float32),
        steps_per_epoch=1,
        epochs=1,
    )
    logger.info("Warm-up training done.")

    dataset_path = Path(dataset_path)

    dataset_name = "KerasSequence"
    logger.info("Start training for {dataset_name}", dataset_name=dataset_name)
    data_loader = loaders.KerasSequence(
        batch_size=batch_size, dataset_path=dataset_path, patch_size=PATCH_SIZE,
    )

    time_dataset(
        model=model,
        dataset=data_loader,
        dataset_name=dataset_name,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )


if __name__ == "__main__":
    run_analysis()
