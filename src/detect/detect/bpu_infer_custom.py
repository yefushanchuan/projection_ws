import cv2
import numpy as np
import bpu_infer_lib

class BPU_Detect:
    def __init__(self, model_path:str,
                labelnames:list,
                num_classes:int = None,
                conf:float = 0.45,
                iou:float = 0.45,
                anchors:np.array = np.array([
                    [10,13, 16,30, 33,23],  # P3/8
                    [30,61, 62,45, 59,119],  # P4/16
                    [116,90, 156,198, 373,326],  # P5/32
                   ]),
                strides = np.array([8, 16, 32]),
                mode:bool = False,
                window_created = False,
                is_save:bool = False
                ):
        self.model = model_path
        self.labelname = labelnames
        self.inf = bpu_infer_lib.Infer(False)
        self.inf.load_model(self.model)
        self.conf = conf
        self.iou = iou
        self.anchors = anchors
        self.strides = strides
        self.input_w = 640
        self.input_h = 640
        self.nc = num_classes if num_classes is not None else len(self.labelname)
        self.mode = mode
        self.window_created = window_created
        self.is_save = is_save
        self._init_grids()

    def _init_grids(self) :
            """初始化特征图网格"""
            def _create_grid(stride: int) :
                """创建单个stride的网格和anchors"""
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
                
            # 创建不同尺度的网格
            self.s_grid, self.s_anchors = _create_grid(self.strides[0])
            self.m_grid, self.m_anchors = _create_grid(self.strides[1]) 
            self.l_grid, self.l_anchors = _create_grid(self.strides[2])
            
            """print(f"网格尺寸: {self.s_grid.shape = }  {self.m_grid.shape = }  {self.l_grid.shape = }")
            print(f"Anchors尺寸: {self.s_anchors.shape = }  {self.m_anchors.shape = }  {self.l_anchors.shape = }")"""


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
        return  nv12
    
    def PreProcess(self, img):#预处理函数
        if isinstance(img, str):
            # 如果输入是字符串 (旧逻辑)，则从文件读取
            orig_img = cv2.imread(img)
            if orig_img is None:
                raise ValueError(f"无法读取图片: {img}")
        elif isinstance(img, np.ndarray):
            # 如果输入是NumPy数组 (新逻辑)，直接使用它
            orig_img = img
        else:
            raise TypeError(f"不支持的输入类型: {type(img)}。请输入 str 或 numpy.ndarray。")

        # 保持比例缩放 + 填充，并返回比例和padding
        def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
            shape = image.shape[:2]  # (h, w)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 缩放比例
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后尺寸 (w, h)

            # 缩放
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

            # 计算填充
            dw = new_shape[1] - new_unpad[0]  # width padding
            dh = new_shape[0] - new_unpad[1]  # height padding
            top, bottom = dh // 2, dh - dh // 2
            left, right = dw // 2, dw - dw // 2

            # 填充
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)

            return padded, r, (left, top)

        # 调用 letterbox
        input_tensor, ratio, (pad_w, pad_h) = letterbox(orig_img, (self.input_h, self.input_w))
        self.ratio = ratio
        self.pad_w = pad_w
        self.pad_h = pad_h  
        # 转换颜色格式
        input_tensor = self.bgr2nv12_opencv(input_tensor)

        # 送入推理
        self.inf.read_input(input_tensor, 0)

    def PostProcess(self, method=1):
        if method == 1 :
            # 方法1：使用get_infer_res_np_float32获取原始输出并处理
            s_pred = self.inf.get_infer_res_np_float32(0)
            m_pred = self.inf.get_infer_res_np_float32(1)
            l_pred = self.inf.get_infer_res_np_float32(2)

            # reshape
            s_pred = s_pred.reshape([-1, (5 + self.nc)])
            m_pred = m_pred.reshape([-1, (5 + self.nc)])
            l_pred = l_pred.reshape([-1, (5 + self.nc)])

            # classify: 利用numpy向量化操作完成阈值筛选
            s_raw_max_scores = np.max(s_pred[:, 5:], axis=1)
            s_max_scores = 1 / ((1 + np.exp(-s_pred[:, 4]))*(1 + np.exp(-s_raw_max_scores)))
            s_valid_indices = np.flatnonzero(s_max_scores >= self.conf)
            s_ids = np.argmax(s_pred[s_valid_indices, 5:], axis=1)
            s_scores = s_max_scores[s_valid_indices]

            m_raw_max_scores = np.max(m_pred[:, 5:], axis=1)
            m_max_scores = 1 / ((1 + np.exp(-m_pred[:, 4]))*(1 + np.exp(-m_raw_max_scores)))
            m_valid_indices = np.flatnonzero(m_max_scores >= self.conf)
            m_ids = np.argmax(m_pred[m_valid_indices, 5:], axis=1)
            m_scores = m_max_scores[m_valid_indices]

            l_raw_max_scores = np.max(l_pred[:, 5:], axis=1)
            l_max_scores = 1 / ((1 + np.exp(-l_pred[:, 4]))*(1 + np.exp(-l_raw_max_scores)))
            l_valid_indices = np.flatnonzero(l_max_scores >= self.conf)
            l_ids = np.argmax(l_pred[l_valid_indices, 5:], axis=1)
            l_scores = l_max_scores[l_valid_indices]

            # 特征解码
            s_dxyhw = 1 / (1 + np.exp(-s_pred[s_valid_indices, :4]))
            s_xy = (s_dxyhw[:, 0:2] * 2.0 + self.s_grid[s_valid_indices,:] - 1.0) * self.strides[0]
            s_wh = (s_dxyhw[:, 2:4] * 2.0) ** 2 * self.s_anchors[s_valid_indices, :]
            s_xyxy = np.concatenate([s_xy - s_wh * 0.5, s_xy + s_wh * 0.5], axis=-1)

            m_dxyhw = 1 / (1 + np.exp(-m_pred[m_valid_indices, :4]))
            m_xy = (m_dxyhw[:, 0:2] * 2.0 + self.m_grid[m_valid_indices,:] - 1.0) * self.strides[1]
            m_wh = (m_dxyhw[:, 2:4] * 2.0) ** 2 * self.m_anchors[m_valid_indices, :]
            m_xyxy = np.concatenate([m_xy - m_wh * 0.5, m_xy + m_wh * 0.5], axis=-1)

            l_dxyhw = 1 / (1 + np.exp(-l_pred[l_valid_indices, :4]))
            l_xy = (l_dxyhw[:, 0:2] * 2.0 + self.l_grid[l_valid_indices,:] - 1.0) * self.strides[2]
            l_wh = (l_dxyhw[:, 2:4] * 2.0) ** 2 * self.l_anchors[l_valid_indices, :]
            l_xyxy = np.concatenate([l_xy - l_wh * 0.5, l_xy + l_wh * 0.5], axis=-1)

            # 大中小特征层阈值筛选结果拼接
            xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
            scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
            ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

            xyxy[:, [0, 2]] -= self.pad_w  # x方向去除pad宽度
            xyxy[:, [1, 3]] -= self.pad_h  # y方向去除pad高度
            xyxy /= self.ratio            # 除以缩放比例还原原图尺寸

        elif method == 0:
            # 方法2：使用get_output获取输出
            if not self.inf.get_output():
                raise RuntimeError("获取输出失败")

            classes_scores = self.inf.outputs[0].data  # (1, 80, 80, 18)
            bboxes = self.inf.outputs[1].data         # (1, 40, 40, 18)

            # 直接使用4D数据
            # 每个网格有3个anchor，每个anchor预测6个值(4个框坐标+1个objectness+1个类别)
            batch, height, width, channels = classes_scores.shape
            num_anchors = 3
            pred_per_anchor = 6

            scores_list = []
            boxes_list = []
            ids_list = []
            # 处理每个网格点
            for h in range(height):
                for w in range(width):
                    for a in range(num_anchors):
                        # 获取当前anchor的预测值
                        start_idx = int(a * pred_per_anchor)
                        box = classes_scores[0, h, w, start_idx:start_idx+4].copy()  # 框坐标
                        obj_score = float(classes_scores[0, h, w, start_idx+4])      # objectness
                        cls_score = float(classes_scores[0, h, w, start_idx+5])      # 类别分数
                        # sigmoid激活
                        obj_score = 1 / (1 + np.exp(-obj_score))
                        cls_score = 1 / (1 + np.exp(-cls_score))
                        score = obj_score * cls_score

                        # 如果分数超过阈值，保存这个预测
                        if score >= self.conf:
                            # 解码框坐标
                            box = 1 / (1 + np.exp(-box))  # sigmoid
                            cx = float((box[0] * 2.0 + w - 0.5) * self.strides[0])
                            cy = float((box[1] * 2.0 + h - 0.5) * self.strides[0])
                            w_pred = float((box[2] * 2.0) ** 2 * self.anchors[0][a*2])
                            h_pred = float((box[3] * 2.0) ** 2 * self.anchors[0][a*2+1])

                            # 转换为xyxy格式
                            x1 = cx - w_pred/2
                            y1 = cy - h_pred/2
                            x2 = cx + w_pred/2
                            y2 = cy + h_pred/2

                            boxes_list.append([x1, y1, x2, y2])
                            scores_list.append(float(score))  # 确保是标量
                            ids_list.append(0)  # 假设只有一个类别
            if boxes_list:
                xyxy = np.array(boxes_list, dtype=np.float32)
                scores = np.array(scores_list, dtype=np.float32)
                ids = np.array(ids_list, dtype=np.int32)
            else:
                xyxy = np.array([], dtype=np.float32).reshape(0, 4)
                scores = np.array([], dtype=np.float32)
                ids = np.array([], dtype=np.int32)
            if len(xyxy) > 0:
                # 还原坐标到原图尺寸
                xyxy[:, [0, 2]] -= self.pad_w
                xyxy[:, [1, 3]] -= self.pad_h
                xyxy /= self.ratio          
        else:
            raise ValueError("method must be 0 or 1")

        # NMS处理
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), self.conf, self.iou)

        if len(indices) > 0:
            indices = np.array(indices).flatten()
            self.bboxes = xyxy[indices].astype(np.int32)
            self.scores = scores[indices]
            self.ids = ids[indices]
            
            self.centers = []
            for (x1, y1, x2, y2) in self.bboxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.centers.append((cx, cy))

            for i, (cx, cy) in enumerate(self.centers):
                print(f"第{i}个框的中心点坐标：({cx}, {cy})")
                
        else:
            print("No detections after NMS")
            self.bboxes = np.array([], dtype=np.int32).reshape(0, 4)
            self.scores = np.array([], dtype=np.float32)
            self.ids = np.array([], dtype=np.int32)

    def draw_detection(self,img: np.array, 
                        box,
                        score: float, 
                        class_id: int,
                        labelname: list):
        x1, y1, x2, y2 = box
        rdk_colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]
        color = rdk_colors[class_id % len(rdk_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{labelname[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(
            img, 
            (label_x, label_y - label_height), 
            (label_x + label_width, label_y + label_height), 
            color, 
            cv2.FILLED
        )
        cv2.putText(img, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def detect_result(self, img):
        if isinstance(img, str):
            draw_img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            draw_img = img.copy()
        else:
            return
        
        for class_id, score, bbox in zip(self.ids, self.scores, self.bboxes):
            x1, y1, x2, y2 = bbox
            self.draw_detection(draw_img, (x1, y1, x2, y2), score, class_id, self.labelname)

        if not self.window_created:
            cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Detection Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.window_created = True

        if self.is_save:
            cv2.imwrite("result.jpg", draw_img)
        else:
            cv2.imshow("Detection Result", draw_img)
            cv2.waitKey(1)  # 非阻塞
  
    def detect(self, img, method_post):
        """
        检测函数
        Args:
            img_path: 图片路径或图片数组
            method_post: 后处理方法
        """
        # 预处理
        self.PreProcess(img)
        
        # 推理和后处理
        self.inf.forward(self.mode)
        self.PostProcess(method_post)
        self.detect_result(img)

if __name__ == "__main__":
    labelname = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    test_img = "../dog.jpeg"
    model_path = "../models/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin"
    infer = BPU_Detect(model_path, labelname, conf = 0.1, mode = False, is_save = True)
    infer.detect(test_img, method_post = 1)