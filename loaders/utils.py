import tensorflow as tf


def select_patch(sharp, blur, patch_size_x, patch_size_y):
    stack = tf.stack([sharp, blur], axis=0)
    patches = tf.image.random_crop(stack, size=[2, patch_size_x, patch_size_y, 3])
    return (patches[0], patches[1])
