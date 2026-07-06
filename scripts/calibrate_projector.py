#!/usr/bin/env python3
import argparse
import time
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ============================================================
# Gray Code generation
# ============================================================

def gray_to_binary(gray, n_bits):
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[..., 0] = gray[..., 0]
    for i in range(1, n_bits):
        binary[..., i] = binary[..., i - 1] ^ gray[..., i]
    return binary


def generate_gray_patterns(width, height, n_col_bits, n_row_bits):
    patterns = []

    # Column indices 0..width-1, Gray: g(x) = x ^ (x >> 1)
    x_col = np.arange(width, dtype=np.uint32)
    gray_col = x_col ^ (x_col >> 1)

    for bit in range(n_col_bits):
        bit_idx = n_col_bits - 1 - bit
        bit_mask = (gray_col >> bit_idx) & 1
        img_direct = np.where(bit_mask.reshape(1, -1) == 1,
                              np.uint8(255), np.uint8(0))
        img_direct = np.tile(img_direct, (height, 1))
        patterns.append(("col", bit, "direct", img_direct))

        img_inverse = 255 - img_direct
        patterns.append(("col", bit, "inverse", img_inverse))

    # Row indices 0..height-1, Gray: g(y) = y ^ (y >> 1)
    y_row = np.arange(height, dtype=np.uint32)
    gray_row = y_row ^ (y_row >> 1)

    for bit in range(n_row_bits):
        bit_idx = n_row_bits - 1 - bit
        bit_mask = (gray_row >> bit_idx) & 1
        img_direct = np.where(bit_mask.reshape(-1, 1) == 1,
                              np.uint8(255), np.uint8(0))
        img_direct = np.tile(img_direct, (1, width))
        patterns.append(("row", bit, "direct", img_direct))

        img_inverse = 255 - img_direct
        patterns.append(("row", bit, "inverse", img_inverse))

    return patterns


# ============================================================
# Gray Code decoding
# ============================================================

def decode_gray_patterns(captured_imgs, n_col_bits, n_row_bits, shadow_threshold):
    n_imgs = len(captured_imgs)
    assert n_imgs == 2 * n_col_bits + 2 * n_row_bits

    H, W = captured_imgs[0].shape[:2]

    col_gray = np.zeros((H, W, n_col_bits), dtype=np.uint8)
    row_gray = np.zeros((H, W, n_row_bits), dtype=np.uint8)
    valid = np.ones((H, W), dtype=bool)

    for i in range(n_col_bits):
        direct = captured_imgs[2 * i].astype(np.float32)
        inverse = captured_imgs[2 * i + 1].astype(np.float32)

        if len(direct.shape) == 3:
            direct = cv2.cvtColor(direct.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            direct = direct.astype(np.float32)
        if len(inverse.shape) == 3:
            inverse = cv2.cvtColor(inverse.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            inverse = inverse.astype(np.float32)

        diff = np.abs(direct - inverse)
        col_gray[:, :, i] = (direct > inverse).astype(np.uint8)
        valid &= (diff > shadow_threshold)

    for i in range(n_row_bits):
        offset = 2 * n_col_bits
        direct = captured_imgs[offset + 2 * i].astype(np.float32)
        inverse = captured_imgs[offset + 2 * i + 1].astype(np.float32)

        if len(direct.shape) == 3:
            direct = cv2.cvtColor(direct.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            direct = direct.astype(np.float32)
        if len(inverse.shape) == 3:
            inverse = cv2.cvtColor(inverse.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            inverse = inverse.astype(np.float32)

        diff = np.abs(direct - inverse)
        row_gray[:, :, i] = (direct > inverse).astype(np.uint8)
        valid &= (diff > shadow_threshold)

    col_binary = gray_to_binary(col_gray, n_col_bits)
    row_binary = gray_to_binary(row_gray, n_row_bits)

    Map_X = col_binary.dot(1 << np.arange(n_col_bits)[::-1]).astype(np.float32)
    Map_Y = row_binary.dot(1 << np.arange(n_row_bits)[::-1]).astype(np.float32)

    return Map_X, Map_Y, valid


def bilinear_interp(map_img, u, v):
    H, W = map_img.shape[:2]

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)

    du, dv = u - u0, v - v0

    f00 = map_img[v0, u0]
    f10 = map_img[v0, u1]
    f01 = map_img[v1, u0]
    f11 = map_img[v1, u1]

    return (1 - du) * (1 - dv) * f00 + du * (1 - dv) * f10 \
           + (1 - du) * dv * f01 + du * dv * f11


# ============================================================
# ROS 2 node — captures camera frames
# ============================================================

class CameraSubscriber(Node):
    def __init__(self, cam_topic):
        super().__init__("calibrate_camera_sub")
        self.bridge = CvBridge()
        self.latest_frame = None
        self.sub = self.create_subscription(
            Image, cam_topic, self._cb, rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info(f"Subscribed to {cam_topic}")

    def _cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def get_frame(self, timeout=5.0):
        started = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            if time.time() - started > timeout:
                return None
        return None

    def flush_and_get(self, n_flush=3, timeout=5.0):
        for _ in range(n_flush):
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.get_frame(timeout)


# ============================================================
# Projector display via OpenCV fullscreen window
# ============================================================

class ProjectorDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = "ProjectorCalib"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, width, height)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, img):
        display = cv2.resize(img, (self.width, self.height))
        cv2.imshow(self.window, display)
        cv2.waitKey(50)

    def white(self):
        img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        self.show(img)

    def close(self):
        cv2.destroyWindow(self.window)


# ============================================================
# Main calibrator
# ============================================================

class ProjectorCalibrator:
    def __init__(self, args):
        self.args = args

        self.K_cam = np.array([[args.cam_fx, 0, args.cam_cx],
                                [0, args.cam_fy, args.cam_cy],
                                [0, 0, 1]], dtype=np.float64)
        self.D_cam = np.array([args.cam_d0, args.cam_d1, args.cam_d2,
                                args.cam_d3, args.cam_d4], dtype=np.float64)

        self.proj_size = (args.proj_width, args.proj_height)
        self.cam_size = None

        n_col_bits = int(np.ceil(np.log2(args.proj_width)))
        n_row_bits = int(np.ceil(np.log2(args.proj_height)))
        self.n_col_bits = n_col_bits
        self.n_row_bits = n_row_bits

        fx_guess = args.proj_width * args.throw_ratio
        self.K_guess = np.array([[fx_guess, 0, args.proj_width / 2.0],
                                  [0, fx_guess, args.proj_height / 2.0],
                                  [0, 0, 1]], dtype=np.float64)

        pattern_size = (args.pattern_cols, args.pattern_rows)
        square_mm = args.square_size
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0],
                                     0:pattern_size[1]].T.reshape(-1, 2) * square_mm

        self.obj_pts_list = []
        self.cam_corners_list = []
        self.proj_corners_list = []

        self.patterns = generate_gray_patterns(
            args.proj_width, args.proj_height,
            n_col_bits, n_row_bits)

        total = len(self.patterns)
        print(f"格雷码条纹: {n_col_bits}bit 列编码 + {n_row_bits}bit 行编码 = {total} 张 (正码+逆码)")

        self.cam_node = CameraSubscriber(args.cam_topic)
        self.projector = ProjectorDisplay(args.proj_width, args.proj_height)

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def capture_pose(self):
        print("  [1/4] 投影白光 + 寻找角点...")
        self.projector.white()
        time.sleep(0.3)
        frame = self.cam_node.flush_and_get(n_flush=3)
        if frame is None:
            print("  ✗ 未能获取相机图像")
            return False

        if self.cam_size is None:
            self.cam_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.args.pattern_cols, self.args.pattern_rows), None)

        if not found:
            print("  ✗ 未找到棋盘格，请调整标定板位置/角度")
            return False

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        cam_corners = corners.reshape(-1, 1, 2)
        print(f"  找到 {len(cam_corners)} 个角点 ✓")

        print(f"  [2/4] 投射格雷码 ({len(self.patterns)} 张)...")
        cap_imgs = []
        t0 = time.time()
        for idx, (ptype, bit, dtype, img) in enumerate(self.patterns):
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.projector.show(img_bgr)
            time.sleep(self.args.sync_delay)
            frame = self.cam_node.get_frame()
            if frame is None:
                print(f"  ✗ 第 {idx} 张格雷码采图失败")
                return False
            cap_imgs.append(frame)
            sys.stdout.write(f"\r  [{idx + 1}/{len(self.patterns)}]")
            sys.stdout.flush()
        elapsed = time.time() - t0
        print(f"  完成 ({elapsed:.1f}s)")

        print("  [3/4] 解码...")
        t0 = time.time()
        Map_X, Map_Y, valid = decode_gray_patterns(
            cap_imgs, self.n_col_bits, self.n_row_bits, self.args.shadow_threshold)
        elapsed = time.time() - t0
        print(f"  解码完成 ({elapsed:.1f}s), 有效像素: {np.sum(valid)}/{valid.size}")

        print("  [4/4] 角点查表 (双线性插值)...")
        corners_proj = np.zeros((len(cam_corners), 1, 2), dtype=np.float32)
        valid_count = 0
        for i, corner in enumerate(cam_corners):
            u_c, v_c = corner[0, 0], corner[0, 1]
            u_p = bilinear_interp(Map_X, u_c, v_c)
            v_p = bilinear_interp(Map_Y, u_c, v_c)
            corners_proj[i, 0, 0] = u_p
            corners_proj[i, 0, 1] = v_p

            vu, vv = int(round(v_c)), int(round(u_c))
            if 0 <= vu < valid.shape[0] and 0 <= vv < valid.shape[1]:
                if valid[vu, vv]:
                    valid_count += 1

        if valid_count < len(cam_corners):
            print(f"  ✗ 有 {len(cam_corners) - valid_count} 个角点在格雷码无效区(阴影/反光)，"
                  f"已丢弃该姿态！")
            return False

        self.obj_pts_list.append(self.objp)
        self.proj_corners_list.append(corners_proj)
        self.cam_corners_list.append(cam_corners)
        print(f"  角点查表完成, 有效角点: {valid_count}/{len(cam_corners)} ✓")

        return True

    def run_calibration(self):
        n = len(self.obj_pts_list)
        if n < 3:
            print("姿态数不足, 至少需要 3 个")
            return

        print(f"\n{'=' * 50}")
        print(f"使用 {n} 个姿态开始标定...")
        print(f"{'=' * 50}")

        obj_pts = [self.objp for _ in range(n)]

        print("\n[阶段一] calibrateCamera → 投影仪内参")
        ret_proj, K_proj, D_proj, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts,
            self.proj_corners_list,
            self.proj_size,
            self.K_guess,
            None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        print(f"  重投影误差 (rms): {ret_proj:.4f} px")

        print("\n[阶段二] stereoCalibrate → 相机-投影仪外参")
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_pts,
            self.cam_corners_list,
            self.proj_corners_list,
            self.K_cam, self.D_cam,
            K_proj, D_proj,
            self.cam_size,
            flags=cv2.CALIB_FIX_INTRINSIC)
        print(f"  双目重投影误差 (rms): {ret_stereo:.4f}")

        print(f"\n{'=' * 50}")
        print(f"标定结果")
        print(f"{'=' * 50}")

        print(f"\nK_proj (投影仪内参):")
        print(f"  fx = {K_proj[0, 0]:.3f}, fy = {K_proj[1, 1]:.3f}")
        print(f"  cx = {K_proj[0, 2]:.3f}, cy = {K_proj[1, 2]:.3f}")
        print(f"\nD_proj (投影仪畸变):")
        print(f"  {np.array2string(D_proj.flatten(), precision=6, suppress_small=True)}")

        print(f"\nR (旋转矩阵, P_proj = R·P_cam + T):")
        for row in R:
            print(f"  [{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}]")
        print(f"\nT (平移向量):")
        print(f"  [{T[0, 0]:.4f}, {T[1, 0]:.4f}, {T[2, 0]:.4f}]")

        print(f"\nK_cam (相机内参, 存档):")
        print(f"  fx = {self.K_cam[0, 0]:.3f}, fy = {self.K_cam[1, 1]:.3f}")
        print(f"  cx = {self.K_cam[0, 2]:.3f}, cy = {self.K_cam[1, 2]:.3f}")

        output = self.args.output
        np.savez(output,
                 K_proj=K_proj, D_proj=D_proj,
                 R=R, T=T,
                 K_cam=self.K_cam, D_cam=self.D_cam,
                 rms_proj=ret_proj, rms_stereo=ret_stereo)
        print(f"\n结果已保存至: {output}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="投影仪-深度相机 联合标定 (灰度码方案)")
    parser.add_argument("--cam-topic", default="/camera/realsense_d435i/color/image_raw")
    parser.add_argument("--cam-fx", type=float, required=True)
    parser.add_argument("--cam-fy", type=float, required=True)
    parser.add_argument("--cam-cx", type=float, required=True)
    parser.add_argument("--cam-cy", type=float, required=True)
    parser.add_argument("--cam-d0", type=float, default=0.0)
    parser.add_argument("--cam-d1", type=float, default=0.0)
    parser.add_argument("--cam-d2", type=float, default=0.0)
    parser.add_argument("--cam-d3", type=float, default=0.0)
    parser.add_argument("--cam-d4", type=float, default=0.0)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=20.0,
                        help="棋盘格大小 (mm)")
    parser.add_argument("--proj-width", type=int, default=1920)
    parser.add_argument("--proj-height", type=int, default=1080)
    parser.add_argument("--throw-ratio", type=float, default=1.26)
    parser.add_argument("--shadow-threshold", type=float, default=30.0,
                        help="格雷码正逆码亮度差阈值 (0-255)")
    parser.add_argument("--sync-delay", type=float, default=0.1,
                        help="格雷码投图-采图同步延迟 (秒)")
    parser.add_argument("--min-poses", type=int, default=15)
    parser.add_argument("--output", default="calib_result.npz")
    args = parser.parse_args()

    rclpy.init()
    calib = ProjectorCalibrator(args)

    print("=" * 50)
    print("投影仪标定程序 — 格雷码稠密匹配方案")
    print("=" * 50)
    print(f"相机内参: fx={args.cam_fx:.3f} fy={args.cam_fy:.3f} "
          f"cx={args.cam_cx:.3f} cy={args.cam_cy:.3f}")
    print(f"棋盘格: {args.pattern_cols}x{args.pattern_rows}, 格子={args.square_size}mm")
    print(f"投影仪: {args.proj_width}x{args.proj_height}, 投射比={args.throw_ratio}")
    print(f"最小姿态: {args.min_poses}")
    print("=" * 50)
    print("操作:")
    print("  Enter = 采集一个姿态")
    print("  c     = 开始标定 (需要 >= {} 个姿态)".format(args.min_poses))
    print("  q     = 退出")
    print("=" * 50)

    try:
        while True:
            n = len(calib.obj_pts_list)
            prompt = f"\n姿态 {n + 1}/{args.min_poses}: " \
                     f"放好棋盘格后按 Enter, q=退出, 已采集={n}"
            if n >= args.min_poses:
                prompt += ", c=开始标定"

            cmd = input(prompt).strip().lower()

            if cmd == "q":
                break
            elif cmd == "c":
                if n < args.min_poses:
                    print(f"姿态不足, 当前 {n}, 需要 >= {args.min_poses}")
                else:
                    calib.run_calibration()
                    break
            elif cmd == "":
                t0 = time.time()
                ok = calib.capture_pose()
                if ok:
                    n = len(calib.obj_pts_list)
                    print(f"  ✓ 采集成功 — 共 {n} 个姿态 ({time.time() - t0:.1f}s)")
            else:
                print("未知命令, 请重试")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        calib.projector.close()
        calib.cam_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
