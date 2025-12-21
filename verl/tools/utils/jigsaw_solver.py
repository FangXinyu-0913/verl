#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw

def _build_rowwise_order_from_centers(
    centers: List[Tuple[float, float]],
    rows_hint: Optional[int] = None,
) -> List[int]:
    """
    根据中心点坐标，将拼图块按“从上到下、从左到右”的顺序排序，返回索引列表。
    """
    n = len(centers)
    if n == 0:
        return []
    ys = [c[1] for c in centers]
    min_y, max_y = min(ys), max(ys)
    
    if rows_hint is None or rows_hint <= 0:
        rows = max(1, int(round(math.sqrt(n))))
    else:
        rows = max(1, rows_hint)

    row_buckets: List[List[int]] = [[] for _ in range(rows)]
    span_y = max_y - min_y
    if span_y < 1e-6:
        return sorted(range(n), key=lambda i: centers[i][0])

    for i, (_, cy) in enumerate(centers):
        t = (cy - min_y) / span_y
        row_idx = int(round(t * (rows - 1)))
        if row_idx < 0:
            row_idx = 0
        if row_idx >= rows:
            row_idx = rows - 1
        row_buckets[row_idx].append(i)

    order: List[int] = []
    for r in range(rows):
        row = row_buckets[r]
        if not row:
            continue
        row.sort(key=lambda i: centers[i][0])  # 同一行按 x 从小到大
        order.extend(row)
    return order

def reassemble_jigsaw(
    puzzle_dir: str,
    order: List[int],
    puzzle_type: str,
    target_resolution: Tuple[int, int],
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
):
    """
    根据给定的顺序将 puzzle_dir 中的碎片重新拼合。
    
    :param puzzle_dir: 包含 pieces 图片和 meta json 的文件夹路径
    :param order: 拼图块 Board ID 的列表。order[i] 表示在恢复后的第 i 个位置（从左上角开始行优先），
                  应该放置 Board ID 为 order[i] 的块。
    :param puzzle_type: 拼图类型 ('rect', 'grid', 'hex')
    :param target_resolution: 目标分辨率 (width, height)
    """
    
    # 1. 根据类型确定 meta json 文件名
    if puzzle_type == "rect":
        meta_filename = "rect_pieces_meta.json"
    elif puzzle_type == "grid":
        meta_filename = "grid_pieces_meta.json"
    elif puzzle_type == "hex":
        meta_filename = "hex_pieces_meta.json"
    else:
        # 尝试自动寻找
        found = False
        for suffix in ["rect", "grid", "hex"]:
            fname = f"{suffix}_pieces_meta.json"
            if os.path.exists(os.path.join(puzzle_dir, fname)):
                meta_filename = fname
                puzzle_type = suffix
                found = True
                break
        if not found:
             # 最后尝试寻找任意 _pieces_meta.json
            for fname in os.listdir(puzzle_dir):
                if fname.endswith("_pieces_meta.json"):
                    meta_filename = fname
                    break
            if not meta_filename:
                raise FileNotFoundError(f"在 {puzzle_dir} 中未找到合适的 *_pieces_meta.json 文件，请检查 --type 参数")

    meta_path = os.path.join(puzzle_dir, meta_filename)
    print(f"[Info] 使用元数据文件: {meta_path}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    width = meta["image_width"]
    height = meta["image_height"]
    pieces_meta = meta["pieces"]
    solution = meta["solution"]  # solution[k] 是正确放置在第 k 个槽位的拼图块的 board_id
    
    # 2. 确定 槽位索引 (Slot Index) 到 pieces 列表索引 (Original Index) 的映射
    # sol_indices[k] = index_in_pieces_list
    # 表示第 k 个视觉槽位对应 meta['pieces'] 中的第几个元素
    
    centers = [(p["center"][0], p["center"][1]) for p in pieces_meta]
    
    if puzzle_type == "hex":
        rings = meta.get("rings", 0)
        rows_hint = 2 * rings - 1 if rings > 0 else None
        sol_indices = _build_rowwise_order_from_centers(centers, rows_hint=rows_hint)
    else:
        # Rect 和 Grid 生成时，pieces 列表已经是按行优先顺序排列
        sol_indices = list(range(len(pieces_meta)))

    # 3. 检查输入 order 的格式 (0-based vs 1-based)
    if not order:
        print("[警告] 输入的 order 为空")
        return None

    # 收集所有合法的 board_id
    valid_board_ids = set(solution)
    
    # 强制 1-based 索引策略：0 视为占位符
    using_zero_based = False
    if 0 in order:
         print("[提示] 检测到 0 (作为空白占位符)，将跳过该位置的填充并绘制轮廓。")
        
    # 4. 创建画布并拼图
    canvas = Image.new("RGBA", (width, height), bg_color)
    draw = ImageDraw.Draw(canvas)
    loaded_images = {}

    # 遍历输入顺序，slot_idx 是重建图的槽位下标 (0, 1, 2...)
    # val 是用户/模型认为该槽位应该放置的 Board ID
    for slot_idx, val in enumerate(order):
        if slot_idx >= len(sol_indices):
            break
            
        target_board_id = val + 1 if using_zero_based else val
        
        # 4.1 确定目标粘贴位置
        # 重建图的第 slot_idx 个位置，对应的 pieces 索引是 sol_indices[slot_idx]
        target_meta_idx = sol_indices[slot_idx]
        target_piece_data = pieces_meta[target_meta_idx]
        target_center = target_piece_data["center"]
        target_cx, target_cy = target_center[0], target_center[1]
        
        # 如果是 0，绘制空白轮廓
        if val == 0:
            # 估算该位置的块大小（使用 bbox）
            # bbox = target_piece_data.get("bbox")
            # if bbox:
            #     bw = bbox[2] - bbox[0]
            #     bh = bbox[3] - bbox[1]
            # else:
            #     # 兜底：均分画布
            #     n_pieces = len(pieces_meta)
            #     bw = width / math.sqrt(n_pieces)
            #     bh = height / math.sqrt(n_pieces)

            # # 统一缩放
            # scale_factor = 0.75
            # final_w = bw * scale_factor
            # final_h = bh * scale_factor

            # # 计算在画布上的绘制中心（假设放在正确位置）
            # # 新坐标 = 画布中心 + (原坐标 - 画布中心) * scale_factor
            # canvas_cx = width / 2.0
            # canvas_cy = height / 2.0
            # final_cx = canvas_cx + (target_cx - canvas_cx) * scale_factor
            # final_cy = canvas_cy + (target_cy - canvas_cy) * scale_factor
            
            # x0 = final_cx - final_w / 2.0
            # y0 = final_cy - final_h / 2.0
            # x1 = final_cx + final_w / 2.0
            # y1 = final_cy + final_h / 2.0
            
            # draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=8)
            continue

        # 4.2 找到 target_board_id 对应的拼图文件

        # 逻辑：
        # solution[k] 表示：在正确解答中，第 k 个槽位放置的是 board_id = solution[k]
        # 因此，board_id = B 的拼图块，原本应该放置在 k = solution.index(B) 这个槽位
        # 这个拼图块的数据位于 pieces[sol_indices[k]]
        
        try:
            # 找到 target_board_id 原本属于哪个槽位 (k)
            original_slot_k = solution.index(target_board_id)
        except ValueError:
            print(f"[跳过] 无法在 solution 中找到 board_id={target_board_id} (slot_idx={slot_idx})")
            return None
            
        # 获取该块的文件信息
        source_meta_idx = sol_indices[original_slot_k]
        source_piece_data = pieces_meta[source_meta_idx]
        filename = source_piece_data["filename"]
        
        # 加载图片
        if filename not in loaded_images:
            p_path = os.path.join(puzzle_dir, filename)
            if not os.path.exists(p_path):
                print(f"[错误] 文件不存在: {p_path}")
                return None
            img = Image.open(p_path).convert("RGBA")
            loaded_images[filename] = img
        
        img = loaded_images[filename]
        
        # 检查是否放置在正确位置
        # slot_idx 是当前位置的逻辑索引 (0, 1, 2...)
        # target_board_id 是预测放置在该位置的 board_id
        # solution[slot_idx] 是该位置真正应该放置的 board_id (从 meta solution 读取)
        
        is_correct = False
        if slot_idx < len(solution):
            true_board_id = solution[slot_idx]
            if target_board_id == true_board_id:
                is_correct = True
        
        # 统一缩放所有拼图块，无论正确与否
        scale_factor = 0.75
        new_w = int(img.width * scale_factor)
        new_h = int(img.height * scale_factor)
        if new_w > 0 and new_h > 0:
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
        
        # 计算粘贴坐标
        if is_correct:
            # 正确位置：将坐标也向画布中心缩放，使得缩小的图片能无缝拼合
            # 新坐标 = 画布中心 + (原坐标 - 画布中心) * scale_factor
            canvas_cx = width / 2.0
            canvas_cy = height / 2.0
            final_cx = canvas_cx + (target_cx - canvas_cx) * scale_factor
            final_cy = canvas_cy + (target_cy - canvas_cy) * scale_factor
        else:
            # 错误位置：保持原坐标不变。因为图片变小了，自然会与周边产生留白
            final_cx = target_cx
            final_cy = target_cy
            
        # 计算粘贴位置：使图片中心对齐目标中心
        paste_x = int(round(final_cx - w / 2.0))
        paste_y = int(round(final_cy - h / 2.0))
        
        canvas.paste(img, (paste_x, paste_y), mask=img)

    if target_resolution:
        canvas.thumbnail(target_resolution)
    return canvas

def main():
    parser = argparse.ArgumentParser(description="拼图重组工具：根据指定顺序重新拼合拼图碎片")
    parser.add_argument("--dir", required=True, help="拼图碎片所在的文件夹路径")
    parser.add_argument("--type", choices=["rect", "grid", "hex"], required=True, help="拼图类型")
    parser.add_argument("--order", required=True, help="拼图顺序。可以是逗号分隔的 Board ID (如 '1,2,3')，或者是包含 JSON 数组的文件路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    
    args = parser.parse_args()
    
    # 解析 order
    order_list = []
    if os.path.isfile(args.order):
        try:
            with open(args.order, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    order_list = [int(x) for x in content]
                else:
                    print("Error: JSON 文件内容必须是列表")
                    return
        except Exception as e:
            print(f"Error reading order file: {e}")
            return
    else:
        # 尝试解析逗号分隔字符串
        try:
            parts = args.order.strip().split(',')
            order_list = [int(x.strip()) for x in parts if x.strip()]
        except ValueError:
            print("Error: order 参数格式不正确，应为逗号分隔的整数或 JSON 文件路径")
            return

    canvas = reassemble_jigsaw(args.dir, order_list, args.type, target_resolution=(448, 448))
    canvas.save(args.output)
    print(f"[成功] 拼图已保存至: {args.output}")

if __name__ == "__main__":
    main()
