from pipelines.mrcnn.pipeline import MrCNNPipeline

# Hide GPU from visible devices
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

image_set_dir = '../if-microscopy-analysis-pipeline/examples/mouse_E16.5/images'
weights_path = "mask_rcnn_lungmap_last.h5"  # change to 'coco' to start from scratch

val_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_002.tif'
test_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001.tif'

mrcnn = MrCNNPipeline(
    image_set_dir=image_set_dir,
    val_img_name=val_img_name,
    test_img_name=test_img_name
)

mrcnn.train(model_weights=weights_path)
