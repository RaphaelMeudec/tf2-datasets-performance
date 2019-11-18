import time
import types
from datetime import datetime
from functools import partial
from pathlib import Path

import click
import tensorflow as tf
from loguru import logger

import loaders
from losses import perceptual_loss
from model import FPNInception

PATCH_SIZE = (256, 256)


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
def time_dataset(model, dataset, dataset_name, steps_per_epoch, epochs, log_dir):
    training_parameters = {
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "callbacks": [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir / dataset_name, profile_batch=3
            )
        ],
    }

    if isinstance(dataset, types.GeneratorType):
        model.fit_generator(dataset, **training_parameters)
    else:
        model.fit(dataset, **training_parameters)


def training(dataset_path, batch_size, epochs, steps_per_epoch, logs_dir, distribute_strategy):

    if distribute_strategy:
        with distribute_strategy.scope():
            vgg = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights="imagenet", input_shape=(*PATCH_SIZE, 3)
            )
            loss_model = tf.keras.models.Model(
                inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
            )
            loss = partial(perceptual_loss, loss_model=loss_model)

            resnet_model = tf.keras.applications.resnet50.ResNet50(
                include_top=False, weights=None, input_shape=(*PATCH_SIZE, 3)
            )
            model = tf.keras.Sequential(
                [resnet_model]
                + [tf.keras.layers.UpSampling2D()] * 5
                + [tf.keras.layers.Conv2D(3, 3, padding="same")]
            )
            optimizer = tf.keras.optimizers.Adam(1e-4)

            model.compile(optimizer=optimizer, loss=loss)
    else:
        vgg = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights="imagenet", input_shape=(*PATCH_SIZE, 3)
        )
        loss_model = tf.keras.models.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
        )
        loss = partial(perceptual_loss, loss_model=loss_model)

        resnet_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights=None, input_shape=(*PATCH_SIZE, 3)
        )
        model = tf.keras.Sequential(
            [resnet_model]
            + [tf.keras.layers.UpSampling2D()] * 5
            + [tf.keras.layers.Conv2D(3, 3, padding="same")]
        )
        optimizer = tf.keras.optimizers.Adam(1e-4)

        model.compile(optimizer=optimizer, loss=loss)

    dataset_path = Path(dataset_path)

    logger.info("Start training for {dataset_name}", dataset_name="")
    data_loader = loaders.PrefetchLoader

    time_dataset(
        model=model,
        dataset=data_loader().load(
            dataset_path, batch_size=batch_size, patch_size=PATCH_SIZE,
        ),
        dataset_name="PrefetchLoader",
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        log_dir=logs_dir,
    )


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
@click.option(
    "--distribute_strategy",
    type=bool,
    default=False,
    help="Whether or not to enable eager execution",
)
def run_analysis(
    epochs, steps_per_epoch, batch_size, dataset_path, distribute_strategy
):
    logs_dir = Path("logs") / str(datetime.timestamp(datetime.now()))
    logger.add(
        logs_dir
        / f"epochs_{epochs}_steps_{steps_per_epoch}_batch_{batch_size}_distribute_{distribute_strategy}.log"
    )
    tf.compat.v1.disable_eager_execution()
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    mirrored_strategy = tf.distribute.MirroredStrategy() if distribute_strategy else None
    training(dataset_path, batch_size, epochs, steps_per_epoch, logs_dir, distribute_strategy=mirrored_strategy)


if __name__ == "__main__":
    run_analysis()
