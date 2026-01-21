import cv2
import numpy as np

# --- 1. 数据结构 (对标 C++ UnifiedResult) ---
class UnifiedResult:
    def __init__(self, class_id, score, box, class_name):
        self.id = int(class_id)
        self.score = float(score)
        self.box = box  # [x, y, w, h] (int)
        self.class_name = str(class_name)
        
        # 计算中心点 (用于深度提取)
        self.center = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))

# --- 2. 数学工具 ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_robust_depth(depth_img, cx, cy):
    """
    提取深度：5x5 区域剔除 0 值后取中值
    """
    if depth_img is None:
        return -1.0
        
    h, w = depth_img.shape
    cx, cy = int(cx), int(cy)
    
    # 边界检查
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return -1.0
    
    x_min = max(0, cx - 2)
    x_max = min(w, cx + 3)
    y_min = max(0, cy - 2)
    y_max = min(h, cy + 3)
    
    roi = depth_img[y_min:y_max, x_min:x_max]
    valid_pixels = roi[roi > 0]
    
    if len(valid_pixels) == 0:
        return -1.0
            
    # 中值滤波，返回单位：米 (原图通常是毫米)
    return float(np.median(valid_pixels)) / 1000.0

# --- 3. 可视化工具 (包含窗口管理) ---
def get_color(class_id):
    # 简单的颜色生成算法，保持和 C++ 一致
    r = (class_id * 123 + 45) % 255
    g = (class_id * 234 + 90) % 255
    b = (class_id * 345 + 135) % 255
    return (b, g, r)

def draw_result(img, result):
    color = get_color(result.id)
    x, y, w, h = result.box
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # 画框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 画标签
    label = f"{result.class_name} {int(result.score * 100)}%"
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1, label_height + 10)
    
    cv2.rectangle(img, (x1, label_y - label_height - 5), (x1 + label_width, label_y + 5), color, -1)
    cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def show_window(win_name, img, results, window_created_dict, fps=0.0):
    """
    统一窗口显示函数
    window_created_dict: 传入一个字典 {'created': bool} 来保存状态 (Python 引用传递技巧)
    """
    draw_img = img.copy()
    
    for res in results:
        draw_result(draw_img, res)
        
    if fps > 0:
        cv2.putText(draw_img, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
    # 窗口复活逻辑
    created = window_created_dict.get('created', False)
    
    # 检查是否被手动关闭
    try:
        if created and cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1.0:
            created = False
    except:
        created = False
        
    if not created:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            created = True
        except:
            pass
            
    if created:
        cv2.imshow(win_name, draw_img)
        cv2.waitKey(1)
        
    window_created_dict['created'] = created