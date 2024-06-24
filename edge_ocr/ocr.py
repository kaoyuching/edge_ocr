import time
import logging
import traceback
from typing import Generator, Iterator, Union, Optional

import numpy as np

from .backends.base import BaseModelInference, MultiModelInference
from .utils.base import InferConfig
from .utils.image_utils import load_detect_image, decode_box, load_ocr_image
from .utils.ocr_utils import ctc_decode, text_add_dash


logger = logging.getLogger(__name__)


class YoloV5OCR(MultiModelInference):
    def __init__(self,
            detect_model: BaseModelInference,
            nms_model: BaseModelInference,
            crnn_model: BaseModelInference,
            infer_config: InferConfig = InferConfig(),
            ):
        super().__init__({'detect': detect_model, 'nms': nms_model, 'crnn': crnn_model})
        self.infer_config = infer_config
        self.LABEL2CHAR = {i + 1: char for i, char in enumerate(self.infer_config.chars)}
        self.default_outputs = (np.zeros([10,10], dtype=np.float32), None, '')

    def run(self, img: Union[str, np.ndarray]):
        # read image for detection
        try:
            input_data, img_orig_shape, img_shape = load_detect_image(img)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.warn(f'Fail to load detect image: {e}. SKIP')
            return self.default_outputs

        # detect car plate
        detect_res = self.models['detect'].run({'images': input_data})['output'] # (1, 16128, 6)
        detect_res = detect_res[0] # single image
        nms_res = self.models['nms'].run({'input': detect_res})['output'] # xywh -> xyxy
        # filter good bbox
        valid_idx, *_ = np.where(nms_res[:, 4] > self.infer_config.bbox_threshold)
        valid_bbox = nms_res[valid_idx, :]
        # rescale back to the original image size
        rescaled_bbox = decode_box(valid_bbox, img_shape, img_orig_shape)

        if len(rescaled_bbox) <= 0:
            return img, *self.default_outputs[1:]

        bbox = rescaled_bbox[0, :4] # xyxy
        # read image for ocr
        try:
            input_data_ocr = load_ocr_image(img, bbox, extend_ratio=1.15)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.warn(f'Fail to load ocr image: {e}. SKIP')
            return img, *self.default_outputs[1:]

        # non-maximum supression
        ocr_pred = self.models['crnn'].run({'input': input_data_ocr})['output'] # (32, 1, 37)
        # ctc decode
        preds = ctc_decode(ocr_pred, beam_size=10, label2char=self.LABEL2CHAR)
        # post process
        text = ''.join(preds[0])
        text = text_add_dash(text)
        if len(text) <= 3:
            text = ''
        text = text.upper()
        return img, bbox.astype(np.int64), text

    def stream(self,
            imgsrc: Union[Generator, Iterator],
            ) -> Generator[np.ndarray, Optional[np.ndarray], str]:
        self.load_models()
        self.activate()
        try:
            for img in imgsrc:
                yield self.run(img)
        finally:
            self.deactivate()
