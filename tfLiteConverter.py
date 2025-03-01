import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL) # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8 # enable LiteRT ops.
#  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops if some ops are not supported by tflite but this won't allow it to be converted to a edgetpu compatible model
]

tflite_model = converter.convert()
# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
