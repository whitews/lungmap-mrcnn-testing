from pipelines.base_pipeline import BasePipeline
from pipelines.mrcnn import model as modellib
from pipelines.mrcnn import visualize
from pipelines.mrcnn.config import Config
from pipelines.common_utils.lungmap_dataset import LungmapDataSet
from pipelines.mrcnn.utils import download_trained_weights
import os
import logging
import numpy as np
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class LungMapTrainingConfig(Config):
    """
    Configuration for training
    """
    NAME = 'lungmap'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classification classes (including background)
    # TODO: update this to be automatic
    NUM_CLASSES = 6  # Override in sub-classes
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    STEPS_PER_EPOCH = 10  # TODO: change to higher number for real run
    TRAIN_ROIS_PER_IMAGE = 300
    VALIDATION_STEPS = 5


DEFAULT_CONFIG = LungMapTrainingConfig()


class MrCNNPipeline(BasePipeline):
    def __init__(self, image_set_dir, val_img_name, test_img_name, model_dir='tmp/models/mrcnn', config=DEFAULT_CONFIG):
        super(MrCNNPipeline, self).__init__(image_set_dir, test_img_name)


        self.config = config

        # TODO: make sure there are enough images in the image set, as this pipeline
        #       needs to separate an image for validation, in addition to the reserved test image.
        self.model_dir = model_dir
        training_data = self.training_data.copy()
        validation_data = {
            val_img_name: training_data.pop(val_img_name)
        }
        test_data = {
            test_img_name: training_data.pop(self.test_img_name)
        }

        self.dataset_train = LungmapDataSet()
        self.dataset_train.load_data_set(training_data)
        self.dataset_train.prepare()

        self.dataset_validation = LungmapDataSet()
        self.dataset_validation.load_data_set(validation_data)
        self.dataset_validation.prepare()

        self.dataset_test = LungmapDataSet()
        self.dataset_test.load_data_set(test_data)
        self.dataset_test.prepare()

        self.model = modellib.MaskRCNN(
            mode="training",
            config=self.config,
            model_dir=self.model_dir
        )

    def train(self, model_weights='coco'):
        """
        Method to train new algorithm
        :param model_weights: required pre-trained h5 file containing model weights or 'coco' to
            download and load COCO model weights
        :return: None
        """
        # Select weights file to load
        if model_weights.lower() == "coco":
            model_weights = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(model_weights):
                download_trained_weights(model_weights)

            self.model.load_weights(
                model_weights,
                by_name=True,
                exclude=[
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask"
                ]
            )
        else:
            self.model.load_weights(
                model_weights,
                by_name=True
            )
        self.model.train(
            self.dataset_train,
            self.dataset_validation,
            learning_rate=self.config.LEARNING_RATE,
            epochs=30,
            layers='heads'
        )
        # self.model.train(
        #     self.dataset_train,
        #     self.dataset_validation,
        #     learning_rate=self.configconfig.LEARNING_RATE / 10,
        #     epochs=2,
        #     layers="all"
        # )

    def _convert_to_contours(self, maskrcnn_dict):
        """
        An internal
        :param maskrcnn_dict:
        :return:
        """
        masks = maskrcnn_dict[0]['masks']
        probs = maskrcnn_dict[0]['scores']
        class_names = [self.dataset_train.class_names[x] for x in maskrcnn_dict[0]['class_ids']]
        final = []
        for i in range(masks.shape[2]):
            instance = {'prob': {class_names[i]: probs[i]}}
            image_8bit = np.uint8(masks[:, :, i].astype('int'))
            contours, hierarchy = cv2.findContours(
                image_8bit,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 1:
                print('Returning more than 1 contour for a mask, what to do?')
                continue
            else:
                instance['contour'] = contours[0]
            final.append(instance)
        return final

    def test(self, model_weights):
        img = self.dataset_test.image_info[0]['img']
        model = modellib.MaskRCNN(
            mode="inference",
            config=self.config,
            model_dir=self.model_dir
        )
        model.load_weights(
            model_weights,
            by_name=True
        )

        results = model.detect([img], verbose=1)
        r = results[0]

        self.test_results = self._convert_to_contours(results)

        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    self.dataset_train.class_names, r['scores'])
