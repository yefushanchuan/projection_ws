import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn #底层

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
        self.model_path = model_path
        self.models = dnn.load(self.model_path)
        self.model = self.models[0]  # 获取第一个模型
        self.labelname = labelnames
        self.conf = conf
        self.iou = iou
        self.anchors = anchors
        self.strides = strides
        self.input_shape = self.model.inputs[0].properties.shape
        self.input_w = self.input_shape[2]  # NCHW格式
        self.input_h = self.input_shape[3]
        self.nc = num_classes if num_classes is not None else len(self.labelname)
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
            orig_img = cv2.imread(img)
            if orig_img is None:
                raise ValueError(f"无法读取图片: {img}")
        elif isinstance(img, np.ndarray):
            orig_img = img
        else:
            raise TypeError(f"不支持的输入类型: {type(img)}。请输入 str 或 numpy.ndarray。")

        def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
            shape = image.shape[:2]  # (h, w)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            dw = new_shape[1] - new_unpad[0]
            dh = new_shape[0] - new_unpad[1]
            top, bottom = dh // 2, dh - dh // 2
            left, right = dw // 2, dw - dw // 2
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)
            return padded, r, (left, top)

        input_tensor, ratio, (pad_w, pad_h) = letterbox(orig_img, (self.input_h, self.input_w))
        self.ratio = ratio
        self.pad_w = pad_w
        self.pad_h = pad_h  
        input_tensor = self.bgr2nv12_opencv(input_tensor)
        return input_tensor

    def PostProcess(self):
        outputs = self.model_outputs
        
        # 处理三个输出层
        s_pred = outputs[0].buffer.reshape([-1, (5 + self.nc)])
        m_pred = outputs[1].buffer.reshape([-1, (5 + self.nc)])
        l_pred = outputs[2].buffer.reshape([-1, (5 + self.nc)])

        # classify
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

        xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        xyxy[:, [0, 2]] -= self.pad_w
        xyxy[:, [1, 3]] -= self.pad_h
        xyxy /= self.ratio

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
        else:
            self.bboxes = np.array([], dtype=np.int32).reshape(0, 4)
            self.scores = np.array([], dtype=np.float32)
            self.ids = np.array([], dtype=np.int32)
            self.centers = []

    def draw_detection(self,img: np.array, box, score: float, class_id: int, labelname: list):
        x1, y1, x2, y2 = box
        rdk_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = rdk_colors[class_id % len(rdk_colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{labelname[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def detect_result(self, img, show_img):
        """
        优化后的逻辑：
        1. 优先处理窗口关闭逻辑。
        2. 如果不需要显示且不需要保存，直接跳过耗时的 copy 和 draw，极大节省 CPU。
        """
        
        # --- 1. 窗口清理逻辑 ---
        # 如果不显示，但之前创建过窗口，则关闭它
        if not show_img:
            if self.window_created:
                try:
                    cv2.destroyWindow("Detection Result")
                except:
                    pass
                self.window_created = False
                cv2.waitKey(1) # 刷新事件循环确保窗口关闭
        
        # --- 2. 提前退出逻辑 (性能优化核心) ---
        # 如果既不保存，也不显示，直接返回，不做任何图像处理
        if not self.is_save and not show_img:
            return

        # ==========================================
        # 只有代码运行到这里，说明要么需要保存，要么需要显示
        # 此时再进行耗时的内存拷贝和绘制操作
        # ==========================================

        # 3. 准备绘制用的图片 (耗时操作: 内存拷贝)
        if isinstance(img, str):
            draw_img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            draw_img = img.copy()
        else:
            return
        
        # 4. 绘制检测框 (耗时操作: 循环绘图)
        for class_id, score, bbox in zip(self.ids, self.scores, self.bboxes):
            x1, y1, x2, y2 = bbox
            self.draw_detection(draw_img, (x1, y1, x2, y2), score, class_id, self.labelname)

        # 5. 保存图片逻辑
        if self.is_save:
            cv2.imwrite("result.jpg", draw_img)
        
        # 6. 显示图片逻辑
        # 注意：这里只需要处理 show_img=True 的情况，False 的情况在第1步已经处理了
        if show_img:
            win_name = "Detection Result"
            
            # 检查窗口是否被用户手动关闭了
            try:
                if self.window_created and cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1.0:
                    self.window_created = False
            except:
                self.window_created = False

            # 如果窗口未创建或已关闭，则重新创建（全屏模式）
            if not self.window_created:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self.window_created = True

            # 显示图像
            cv2.imshow(win_name, draw_img)
            cv2.waitKey(1)
  
    def detect(self, img, show_img=True):
        input_tensor = self.PreProcess(img)
        self.model_outputs = self.model.forward(input_tensor)
        self.PostProcess()
        self.detect_result(img, show_img)