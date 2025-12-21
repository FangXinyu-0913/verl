import cv2
import numpy as np
import argparse
import os
import json

def get_segment(pts, idx1, idx2):
    """Return points from idx1 to idx2 (inclusive) handling wrapping."""
    if idx1 <= idx2:
        return pts[idx1 : idx2 + 1]
    else:
        return np.vstack((pts[idx1:], pts[:idx2 + 1]))

def analyze_piece_for_rect(image_path):
    # {
    #     "top": "flat",
    #     "right": "flat",
    #     "bottom": "innie",
    #     "left": "outie"
    # }
    if not os.path.exists(image_path):
        return {"error": f"File {image_path} not found"}

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "Failed to load image"}
    
    # 1. Extract Contour
    # Use alpha channel to find the shape
    alpha = img[:, :, 3]
    _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Use CHAIN_APPROX_NONE to preserve all points for accurate deviation analysis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"error": "No contour found"}
        
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    n_pts = len(pts)

    # 2. Corner Detection using approxPolyDP
    perimeter = cv2.arcLength(cnt, True)
    
    # Start with a high epsilon to find the rough quadrilateral shape.
    # We decrease epsilon until we find at least 4 corners.
    epsilon_factor = 0.05
    final_approx = None
    
    for _ in range(20):
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4:
            final_approx = approx
            break
        epsilon_factor *= 0.8
        
    corner_indices = []
    if final_approx is not None and len(final_approx) >= 4:
        # Map approx points to contour indices
        approx_pts = final_approx[:, 0, :]
        candidates = []
        for apt in approx_pts:
            dists = np.sum((pts - apt)**2, axis=1)
            idx = np.argmin(dists)
            candidates.append(idx)
        
        # If > 4 points found, pick the 4 that are closest to the Bounding Box corners
        # This helps ignore points detected on large tabs.
        if len(candidates) > 4:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_corners = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]]) # TL, TR, BR, BL
            
            selected_indices = []
            for bc in bbox_corners:
                best_dist = float('inf')
                best_idx = -1
                for c_idx in candidates:
                    dist = np.linalg.norm(pts[c_idx] - bc)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c_idx
                if best_idx not in selected_indices:
                    selected_indices.append(best_idx)
            corner_indices = selected_indices
        else:
            corner_indices = candidates
    else:
         # Fallback to BBox closest points if poly approx fails
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_corners = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
        corner_indices = []
        for bc in bbox_corners:
            dists = np.sum((pts - bc)**2, axis=1)
            corner_indices.append(np.argmin(dists))
            
    if len(corner_indices) < 4:
         return {"error": "Could not detect 4 corners"}

    # 3. Sort Corners: TL, TR, BR, BL
    corners_pts = pts[corner_indices]
    
    # Sort by Y (Top vs Bottom)
    corners_pts_indices = sorted(range(len(corners_pts)), key=lambda i: corners_pts[i][1])
    top_indices = corners_pts_indices[:2]
    bot_indices = corners_pts_indices[2:]
    
    # Sort Top by X (Left vs Right)
    if corners_pts[top_indices[0]][0] < corners_pts[top_indices[1]][0]:
        tl_local_idx, tr_local_idx = top_indices[0], top_indices[1]
    else:
        tl_local_idx, tr_local_idx = top_indices[1], top_indices[0]
        
    # Sort Bot by X
    if corners_pts[bot_indices[0]][0] < corners_pts[bot_indices[1]][0]:
        bl_local_idx, br_local_idx = bot_indices[0], bot_indices[1]
    else:
        bl_local_idx, br_local_idx = bot_indices[1], bot_indices[0]
        
    tl = corner_indices[tl_local_idx]
    tr = corner_indices[tr_local_idx]
    br = corner_indices[br_local_idx]
    bl = corner_indices[bl_local_idx]
    
    corner_map = {"tl": tl, "tr": tr, "br": br, "bl": bl}
    
    # 4. Calculate Centroid
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00']) if M['m00']!=0 else 0
    cy = int(M['m01']/M['m00']) if M['m00']!=0 else 0
    centroid = np.array([cx, cy])

    results = {}
    edges = [
        ("top", "tl", "tr"),
        ("right", "tr", "br"),
        ("bottom", "br", "bl"),
        ("left", "bl", "tl")
    ]
    
    for name, c1_name, c2_name in edges:
        idx1 = corner_map[c1_name]
        idx2 = corner_map[c2_name]
        
        # Determine shortest path along contour between the two corners
        if idx1 <= idx2:
            len1 = idx2 - idx1
        else:
            len1 = (n_pts - idx1) + idx2
        if idx2 <= idx1:
            len2 = idx1 - idx2
        else:
            len2 = (n_pts - idx2) + idx1
            
        if len1 < len2:
            segment = get_segment(pts, idx1, idx2)
        else:
            segment = get_segment(pts, idx2, idx1)
            
        # 5. Analyze Edge Deviation
        if len(segment) < 10:
            results[name] = "flat"
            continue
            
        p1 = segment[0]
        p2 = segment[-1]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        
        # Compute deviation of segment points from the straight line (chord)
        cross = np.cross(vec, segment - p1) / length
        max_abs = np.max(np.abs(cross))
        
        # Threshold: 5% of edge length, with a minimum of 10px
        threshold = max(10.0, length * 0.05)
        
        if max_abs < threshold:
            results[name] = "flat"
        else:
            # Determine Innie or Outie
            min_dev = np.min(cross)
            max_dev = np.max(cross)
            
            if abs(max_dev) > abs(min_dev):
                extreme_idx = np.argmax(cross)
            else:
                extreme_idx = np.argmin(cross)
            
            extreme_pt = segment[extreme_idx]
            midpoint = (p1 + p2) / 2
            
            # Distance comparison relative to centroid
            d_ext = np.linalg.norm(extreme_pt - centroid)
            d_mid = np.linalg.norm(midpoint - centroid)
            
            if d_ext > d_mid:
                results[name] = "outie"
            else:
                results[name] = "innie"

    return results
    # results format


import cv2
import numpy as np
import math
import argparse
import os
import json
from tqdm import tqdm

def calculate_bearing(cx, cy, x, y):
    """
    计算点(x,y)相对于中心(cx,cy)的角度。
    标准：12点钟方向为0度，顺时针为正 (0-360)。
    """
    dx = x - cx
    dy = y - cy
    # atan2(y, x) 默认是 X轴正向为0，逆时针。
    # 我们需要：Y轴负向(上方)为0，顺时针。
    # 数学变换：
    # standard_angle = atan2(dy, dx) (y向下为正)
    # 转换到时钟坐标：degrees = (math.degrees(atan2(dx, -dy)) + 360) % 360
    # dx, -dy 是为了把Y轴翻转指向上方
    angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    return angle

def get_contour_point_at_angle(contour, center, target_angle, tolerance=5.0):
    """
    在轮廓上找到最接近指定角度的点。
    如果在 tolerance 范围内有多个点，取距离中心最远的一个（通常是顶点）。
    """
    cx, cy = center
    best_pt = None
    min_angle_diff = float('inf')
    max_dist = -1

    # 优化：可以先建立索引，这里为了简单直接遍历
    # 实际轮廓点很多，找最接近角度的
    candidates = []
    
    for pt in contour:
        p = pt[0]
        ang = calculate_bearing(cx, cy, p[0], p[1])
        
        # 处理 0/360 的边界问题
        diff = abs(ang - target_angle)
        if diff > 180:
            diff = 360 - diff
            
        if diff < min_angle_diff:
            min_angle_diff = diff
            best_pt = p
            max_dist = math.hypot(p[0]-cx, p[1]-cy)
            candidates = [p]
        elif abs(diff - min_angle_diff) < 0.5: # 角度非常接近
            candidates.append(p)
            
    # 在角度最接近的一组点中，选距离最远的（顶点通常是凸出的）
    final_pt = best_pt
    curr_max_dist = -1
    for p in candidates:
        d = math.hypot(p[0]-cx, p[1]-cy)
        if d > curr_max_dist:
            curr_max_dist = d
            final_pt = p
            
    return final_pt

def analyze_hex_piece_angular(image_path, debug=False):
    # {
    #     "0 - 60": "flat",
    #     "60 - 120": "innie",
    #     "120 - 180": "outie",
    #     "180 - 240": "outie",
    #     "240 - 300": "flat",
    #     "300 - 360": "flat"
    # }
    # 1. 读取与预处理
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return

    alpha = img[:, :, 3]
    _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No contour found")
        return
    cnt = max(contours, key=cv2.contourArea)

    # 2. 计算重心
    M = cv2.moments(cnt)
    if M["m00"] == 0: return
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    debug_img = img.copy() if debug else None
    if debug:
        cv2.circle(debug_img, center, 5, (0, 255, 255, 255), -1)

    edges_status = {}
    
    print(f"Analysing: {os.path.basename(image_path)}")
    print(f"{'Edge Range':<15} | {'Status':<10} | {'Details (In/Out Dist)'}")
    print("-" * 60)

    # 3. 按 60 度扇区遍历
    for i in range(6):
        start_angle = i * 60
        end_angle = (i + 1) * 60
        
        # 找到该扇区的起止点 (即六边形的两个顶点)
        p_start = get_contour_point_at_angle(cnt, center, start_angle)
        p_end = get_contour_point_at_angle(cnt, center, end_angle if end_angle < 360 else 0)
        
        if p_start is None or p_end is None:
            edges_status[f"{start_angle} - {end_angle}"] = "unknown"
            continue
            
        # 4. 分析该扇区内的所有轮廓点
        # 建立基准线向量 p_start -> p_end
        # 直线一般式: Ax + By + C = 0
        A = p_start[1] - p_end[1]
        B = p_end[0] - p_start[0]
        C = p_start[0] * p_end[1] - p_end[0] * p_start[1]
        denominator = math.hypot(A, B)
        if denominator == 0: denominator = 1e-5
        
        # 计算中心到直线的有向距离，作为参考符号
        # dist_center = (A*cx + B*cy + C) / denominator
        # 我们定义：如果点和中心在直线同侧 -> Innie (距离符号相同)
        # 如果点和中心在直线异侧 -> Outie (距离符号相反)
        val_center = A * cx + B * cy + C
        
        max_outie_dist = 0
        max_innie_dist = 0
        
        # 收集该角度范围内的轮廓点
        pts_in_sector = []
        for pt in cnt:
            p = pt[0]
            ang = calculate_bearing(cx, cy, p[0], p[1])
            
            # 简单的角度包含判断
            in_range = False
            if start_angle < end_angle:
                if start_angle <= ang <= end_angle: in_range = True
            else: # 跨越 360 (例如 300-0) - 本例中0-60/60-120不会触发，除非逻辑变动
                if ang >= start_angle or ang <= (end_angle % 360): in_range = True
            
            if in_range:
                val_p = A * p[0] + B * p[1] + C
                dist = abs(val_p) / denominator
                
                # 判断同侧异侧
                # 浮点数积小于0 -> 异号 -> 异侧 -> Outie
                if val_p * val_center < 0:
                    max_outie_dist = max(max_outie_dist, dist)
                else:
                    max_innie_dist = max(max_innie_dist, dist)

        # 5. 判定逻辑
        # 阈值：边长的 8% 左右。边长约等于 p_start 到 p_end 的距离
        edge_len = math.hypot(p_start[0]-p_end[0], p_start[1]-p_end[1])
        threshold = edge_len * 0.08 
        
        status = "flat"
        # 优先级：如果有显著的 outie，就是 outie (因为 innie 也会有边缘点接近直线，但 outie 肯定有很远的点)
        # 但是，innie 同样会有很远的内凹点。
        # 需要看谁更显著。Flat 的话两个都会很小。
        
        if max_outie_dist > threshold and max_outie_dist > max_innie_dist:
            status = "outie"
        elif max_innie_dist > threshold and max_innie_dist > max_outie_dist:
            status = "innie"
        # 某些复杂情况（例如波浪形）可能两者都大，这里简化处理，取最大者
        
        edges_status[f"{start_angle} - {end_angle}"] = status
        
        print(f"{start_angle:3d}° - {end_angle:3d}° | {status.upper():<10} | Out:{max_outie_dist:.1f}, In:{max_innie_dist:.1f}, Thres:{threshold:.1f}")

        if debug:
            # 画基准线
            cv2.line(debug_img, tuple(p_start), tuple(p_end), (255, 0, 0, 255), 2)
            # 标注文字
            mid_x = (p_start[0] + p_end[0]) // 2
            mid_y = (p_start[1] + p_end[1]) // 2
            
            label = status[0].upper()
            color = (0, 0, 255, 255) if status == "outie" else ((0, 255, 0, 255) if status == "innie" else (200, 200, 200, 255))
            
            #稍微往外偏移一点以便看清
            dir_x = mid_x - cx
            dir_y = mid_y - cy
            l = math.hypot(dir_x, dir_y)
            off_x = int(mid_x + (dir_x/l)*20)
            off_y = int(mid_y + (dir_y/l)*20)
            
            cv2.putText(debug_img, label, (off_x, off_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if debug:
        out_file = "debug_angular_" + os.path.basename(image_path)
        cv2.imwrite(out_file, debug_img)
        print(f"\nDebug image saved to {out_file}")

    return edges_status


def analyze_edges_for_piece(image_path, type):
    if type == "rect":
        return analyze_piece_for_rect(image_path)
    elif type == "hex":
        return analyze_hex_piece_angular(image_path)
    else:
        raise ValueError(f"Invalid piece type: {type}")
