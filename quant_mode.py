import tensorflow_model_optimization as tfmot
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.

def lite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

def quant_model(model):
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    

    iou = sm.metrics.IOUScore(threshold=0.5)
    fscore = sm.metrics.FScore(threshold=0.5)
    metrics = [iou, fscore]

    optimizer = Adam(1e-3)
    q_aware_model.compile(optimizer=optimizer, loss=total_loss,
        metrics=metrics)
    return q_aware_model

