import tensorflow as tf


def keras_to_tflite(model, path, fp16=False):
    """Convert a Keras model to .tflite. Built-in ops only, never select-TF-ops."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    with open(path, 'wb') as f:
        f.write(conv.convert())


def function_to_tflite(fn, specs, path):
    """For engines. specs: list of named tf.TensorSpec, all float32."""
    concrete = tf.function(fn, input_signature=specs).get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], tf.Module())
    with open(path, 'wb') as f:
        f.write(conv.convert())
