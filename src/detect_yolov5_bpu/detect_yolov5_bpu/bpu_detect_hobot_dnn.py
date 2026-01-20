import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
# 假设上面的 utils.py 在同级目录 detect_yolov5_bpu 包中
from detect_yolov5_bpu.utils import UnifiedResult

class BPU_Detect:
    def __init__(self, model_path:str,
                labelnames:list,
                num_classes:int = None,
                conf:float = 0.45,
                iou:float = 0.45):
        
        self.model_path = model_path
        self.models = dnn.load(self.model_path)
        self.model = self.models[0]
        self.labelname = labelnames
        self.conf = conf
        self.iou = iou
        
        # 默认 Anchors (YOLOv5)
        self.anchors = np.array([
            [10,13, 16,30, 33,23],       # P3/8
            [30,61, 62,45, 59,119],      # P4/16
            [116,90, 156,198, 373,326],  # P5/32
        ])
        self.strides = np.array([8, 16, 32])
        
        self.input_shape = self.model.inputs[0].properties.shape
        self.input_w = self.input_shape[2]
        self.input_h = self.input_shape[3]
        self.nc = num_classes if num_classes is not None else len(self.labelname)
        
        self._init_grids()

    def _init_grids(self):
        def _create_grid(stride):
            grid = np.stack([
                np.tile(np.linspace(0.5, self.input_w//stride - 0.5, self.input_w//stride), 
                    reps=self.input_h//stride),
                np.repeat(np.arange(0.5, self.input_h//stride + 0.5, 1), 
                        self.input_w//stride)
            ], axis=0).transpose(1,0)
            grid = np.hstack([grid] * 3).reshape(-1, 2)
            
            anchors = np.tile(
                self.anchors[int(np.log2(stride/8))], 
                self.input_w//stride * self.input_h//stride
            ).reshape(-1, 2)
            return grid, anchors
            
        self.s_grid, self.s_anchors = _create_grid(self.strides[0])
        self.m_grid, self.m_anchors = _create_grid(self.strides[1]) 
        self.l_grid, self.l_anchors = _create_grid(self.strides[2])

    def bgr2nv12_opencv(self, image):
        height, width = image.shape[0], image.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
    
    def PreProcess(self, img):
        if isinstance(img, str):
            orig_img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            orig_img = img
        else:
            return None

        # Letterbox logic
        shape = orig_img.shape[:2]  # (h, w)
        r = min(self.input_h / shape[0], self.input_w / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        resized = cv2.resize(orig_img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        dw = self.input_w - new_unpad[0]
        dh = self.input_h - new_unpad[1]
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 保存用于后处理还原坐标的参数
        self.ratio = r
        self.pad_w = left
        self.pad_h = top  
        
        return self.bgr2nv12_opencv(padded)

    def detect(self, img):
        """
        核心推理函数
        :param img: 输入图像 (BGR numpy array)
        :return: List[UnifiedResult]
        """
        input_tensor = self.PreProcess(img)
        if input_tensor is None:
            return []

        # Forward
        outputs = self.model.forward(input_tensor)
        
        # --- PostProcess Logic ---
        s_pred = outputs[0].buffer.reshape([-1, (5 + self.nc)])
        m_pred = outputs[1].buffer.reshape([-1, (5 + self.nc)])
        l_pred = outputs[2].buffer.reshape([-1, (5 + self.nc)])

        def process_layer(pred, grid, anchors, stride):
            raw_max_scores = np.max(pred[:, 5:], axis=1)
            max_scores = 1 / ((1 + np.exp(-pred[:, 4]))*(1 + np.exp(-raw_max_scores)))
            valid_indices = np.flatnonzero(max_scores >= self.conf)
            
            if len(valid_indices) == 0:
                return np.empty((0,4)), np.empty(0), np.empty(0, dtype=int)

            ids = np.argmax(pred[valid_indices, 5:], axis=1)
            scores = max_scores[valid_indices]
            
            dxyhw = 1 / (1 + np.exp(-pred[valid_indices, :4]))
            xy = (dxyhw[:, 0:2] * 2.0 + grid[valid_indices,:] - 1.0) * stride
            wh = (dxyhw[:, 2:4] * 2.0) ** 2 * anchors[valid_indices, :]
            xyxy = np.concatenate([xy - wh * 0.5, xy + wh * 0.5], axis=-1)
            return xyxy, scores, ids

        s_xyxy, s_scores, s_ids = process_layer(s_pred, self.s_grid, self.s_anchors, self.strides[0])
        m_xyxy, m_scores, m_ids = process_layer(m_pred, self.m_grid, self.m_anchors, self.strides[1])
        l_xyxy, l_scores, l_ids = process_layer(l_pred, self.l_grid, self.l_anchors, self.strides[2])

        xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # 还原坐标 (x - pad) / ratio
        if len(xyxy) > 0:
            xyxy[:, [0, 2]] -= self.pad_w
            xyxy[:, [1, 3]] -= self.pad_h
            xyxy /= self.ratio

        # NMS
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), self.conf, self.iou)

        results = []
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            final_boxes = xyxy[indices].astype(np.int32)
            final_scores = scores[indices]
            final_ids = ids[indices]

            for i in range(len(indices)):
                x1, y1, x2, y2 = final_boxes[i]
                
                # 构造 UnifiedResult
                # 注意：UnifiedResult 期望 box 为 [x, y, w, h]
                box_xywh = [x1, y1, x2 - x1, y2 - y1]
                
                class_name = self.labelname[final_ids[i]] if final_ids[i] < len(self.labelname) else "unknown"
                
                res = UnifiedResult(
                    class_id=final_ids[i],
                    score=final_scores[i],
                    box=box_xywh,
                    class_name=class_name
                )
                results.append(res)
                
        return results