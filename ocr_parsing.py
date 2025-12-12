import os
import re
import json
import argparse
from typing import List, Tuple

# ----------------------------
# Geometry helpers
# ----------------------------
def bbox_intersection_area(b1: Tuple[int, int, int, int],
                           b2: Tuple[int, int, int, int]) -> int:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)

def bbox_area(b: Tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

_INT_RE = re.compile(r"-?\d+")
_TD_WITH_BBOX_RE = re.compile(
    r"(<td\b[^>]*\bbbox\s*=\s*\"([^\"]+)\"[^>]*>)(.*?)(</td>)",
    re.IGNORECASE | re.DOTALL
)

def parse_bbox_from_string(s: str) -> Tuple[int, int, int, int]:
    nums = _INT_RE.findall(s)
    if len(nums) < 4:
        raise ValueError(f"Could not parse 4 integers from bbox string: {s!r}")
    return tuple(map(int, nums[:4]))

## mapping layout.txt and word.txt

def load_words_for_folder(folder_path: str) -> List[Tuple[Tuple[int,int,int,int], str]]:
    layout_path = os.path.join(folder_path, "layout.txt")
    ocr_path = os.path.join(folder_path, "ocr.txt")

    if not os.path.isfile(layout_path) or not os.path.isfile(ocr_path):
        raise FileNotFoundError(f"Missing layout.txt or ocr.txt in {folder_path}")

    with open(layout_path, "r", encoding="utf-8") as f:
        layout_lines = [ln.strip() for ln in f if ln.strip()]
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_lines = [ln.rstrip("\n") for ln in f]

    if len(layout_lines) != len(ocr_lines):
        min_len = min(len(layout_lines), len(ocr_lines))
        print(f"[WARN] {folder_path}: layout({len(layout_lines)}) != ocr({len(ocr_lines)}). Truncating to {min_len}.")
        layout_lines = layout_lines[:min_len]
        ocr_lines = ocr_lines[:min_len]

    words = []
    for i, (bl, w) in enumerate(zip(layout_lines, ocr_lines)):
        try:
            bbox = parse_bbox_from_string(bl)
        except Exception as e:
            print(f"[WARN] Skipping invalid bbox line #{i} in {folder_path}: {bl!r} ({e})")
            continue
        words.append((bbox, w))
    return words

## delta html loading from json

def load_result_html_from_folder(folder_path: str) -> str:
    for candidate in ("result.json", "out.json"):
        p = os.path.join(folder_path, candidate)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            if isinstance(data, str):
                return data
    for candidate in ("result.html", "out.html"):
        p = os.path.join(folder_path, candidate)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError("No result.json/out.json/result.html found in folder")

# ----------------------------
# Step 1 – Strip all td inner content
# ----------------------------
def clear_td_contents(html: str) -> str:
    """Remove all inner text of <td> while preserving bbox + structure"""
    def _repl(m: re.Match) -> str:
        return f"{m.group(1)}{m.group(4)}"
    return _TD_WITH_BBOX_RE.sub(_repl, html)

# ----------------------------
# Step 2 – Fill td with OCR words by bbox overlap
# ----------------------------
def debug_fill_td_cells_with_words(html: str,
                             words: List[Tuple[Tuple[int,int,int,int], str]],
                             overlap_threshold: float = 0.0001) -> str:
    out_parts = []
    last_end = 0

    for m in _TD_WITH_BBOX_RE.finditer(html):
        out_parts.append(html[last_end:m.start()])

        open_tag = m.group(1)
        bbox_str = m.group(2)
        close_tag = m.group(4)

        try:
            ## bhashini bbox are 5 values in tuple we only need 1st four
            block_bbox = parse_bbox_from_string(bbox_str)
        except Exception as e:
            print(f"[WARN] Could not parse td bbox {bbox_str!r}: {e}")
            out_parts.append(f"{open_tag}{close_tag}")
            last_end = m.end()
            continue

        matched = []
        for (wb, word) in words:
            inter = bbox_intersection_area(wb, block_bbox)
            area = bbox_area(wb)
            ratio = inter / area if area > 0 else 0.0
            if ratio > overlap_threshold:
                matched.append((wb, word))

        matched.sort(key=lambda w: (w[0][1], w[0][0]))
        matched_words = [w for _, w in matched]
        new_inner = " ".join(matched_words)

        out_parts.append(f"{open_tag}{new_inner}{close_tag}")
        last_end = m.end()

    out_parts.append(html[last_end:])
    return "".join(out_parts)

# ----------------------------
# printing word and bbox from layout.txt and ocr.txt
# ----------------------------
def pretty_print_mapping(words):
    print("\nWord <-> BBox mapping:")
    for i, (bbox, word) in enumerate(words):
        print(f"  #{i:03d}: {word!r} -> {bbox}")
    print(f"Total words: {len(words)}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to OCR folder")
    parser.add_argument("--overlap", type=float, default=0.00001)
    args = parser.parse_args()

    words = load_words_for_folder(args.folder)
    pretty_print_mapping(words)

    html = '''<html><table border="1" class="ocr_tab" title=""><tbody>
<tr><td bbox="36 11 479 112" title="bbox 36 11 479 112">संकेतक</td>
<td bbox="501 11 787 112" title="bbox 501 11 787 112">औसत*</td>
<td bbox="799 11 1454 112" title="bbox 799 11 1454 112">सबसे अच्छा प्रदर्शन</td>
<td bbox="1445 11 2007 112" title="bbox 1445 11 2007 112">सबसे खराब प्रदर्शन</td></tr>
<tr><td bbox="36 122 479 377" title="bbox 36 122 479 377">स्टंटेड बच्चे</td>
<td bbox="501 122 787 377" title="bbox 501 122 787 377">38.7%</td>
<td bbox="799 122 1454 377" title="bbox 799 122 1454 377">केरलः १९.४%</td>
<td bbox="1445 122 2007 377" title="bbox 1445 122 2007 377">उत्तर प्रदेशः ५०.४%</td></tr>
</tbody></table></html>'''

    print("\nOriginal HTML snippet:\n", html[:300])

    cleared_html = clear_td_contents(html)
    print("\nHTML with td cleared:\n", cleared_html[:300])

    final_html = debug_fill_td_cells_with_words(cleared_html, words, overlap_threshold=args.overlap)
    print("\nFinal HTML with filled words (snippet):\n", final_html)

if __name__ == "__main__":
    main()



# # import os
# # import re
# # import json
# # from typing import List, Tuple, Dict, Any

# # # ----------------------------
# # # Geometry helpers
# # # ----------------------------
# # def bbox_intersection_area(b1: Tuple[int, int, int, int],
# #                            b2: Tuple[int, int, int, int]) -> int:
# #     x1 = max(b1[0], b2[0])
# #     y1 = max(b1[1], b2[1])
# #     x2 = min(b1[2], b2[2])
# #     y2 = min(b1[3], b2[3])
# #     if x2 <= x1 or y2 <= y1:
# #         return 0
# #     return (x2 - x1) * (y2 - y1)

# # def bbox_area(b: Tuple[int, int, int, int]) -> int:
# #     return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

# # def get_words_in_block(
# #     block_bbox: Tuple[int, int, int, int],
# #     words: List[Tuple[Tuple[int, int, int, int], str]],
# #     overlap_threshold: float = 0.1
# # ) -> List[str]:
# #     assigned = []
# #     for word_bbox, word in words:
# #         word_area = bbox_area(word_bbox)
# #         inter_area = bbox_intersection_area(word_bbox, block_bbox)
# #         overlap_ratio = inter_area / word_area if word_area > 0 else 0
# #         if overlap_ratio > overlap_threshold:
# #             assigned.append((word_bbox, word))
# #     assigned.sort(key=lambda w: (w[0][1], w[0][0]))
# #     return [w[1] for w in assigned]

# # # ----------------------------
# # # Parsing helpers
# # # ----------------------------
# # _INT_RE = re.compile(r"-?\d+")
# # _TD_WITH_BBOX_RE = re.compile(
# #     r"(<td\b[^>]*\bbbox\s*=\s*\"([^\"]+)\"[^>]*>)(.*?)(</td>)",
# #     re.IGNORECASE | re.DOTALL
# # )

# # def parse_bbox_from_string(s: str) -> Tuple[int, int, int, int]:
# #     nums = _INT_RE.findall(s)
# #     if len(nums) < 4:
# #         raise ValueError(f"Could not parse 4 integers from bbox string: {s!r}")
# #     return tuple(map(int, nums[:4]))

# # def parse_bbox_line(line: str) -> Tuple[int, int, int, int]:
# #     return parse_bbox_from_string(line)

# # # ----------------------------
# # # IO helpers
# # # ----------------------------
# # def load_words_for_folder(folder_path: str) -> List[Tuple[Tuple[int, int, int, int], str]]:
# #     layout_path = os.path.join(folder_path, "layout.txt")
# #     ocr_path = os.path.join(folder_path, "ocr.txt")

# #     if not os.path.isfile(layout_path) or not os.path.isfile(ocr_path):
# #         raise FileNotFoundError(f"Missing layout.txt or ocr.txt in {folder_path}")

# #     with open(layout_path, "r", encoding="utf-8") as f:
# #         layout_lines = [ln.strip() for ln in f if ln.strip()]
# #     with open(ocr_path, "r", encoding="utf-8") as f:
# #         ocr_lines = [ln.rstrip("\n") for ln in f]

# #     if len(layout_lines) != len(ocr_lines):
# #         min_len = min(len(layout_lines), len(ocr_lines))
# #         print(f"[WARN] {folder_path}: layout({len(layout_lines)}) != ocr({len(ocr_lines)}). Truncating to {min_len}.")
# #         layout_lines = layout_lines[:min_len]
# #         ocr_lines = ocr_lines[:min_len]

# #     words = []
# #     for bbox_line, word in zip(layout_lines, ocr_lines):
# #         try:
# #             bbox = parse_bbox_line(bbox_line)
# #         except Exception as e:
# #             print(f"[WARN] Skipping invalid bbox in {folder_path}: {bbox_line!r} ({e})")
# #             continue
# #         words.append((bbox, word))
# #     return words

# # # ----------------------------
# # # HTML transformation
# # # ----------------------------
# # def fill_td_cells_with_words(html: str,
# #                              words: List[Tuple[Tuple[int, int, int, int], str]],
# #                              overlap_threshold: float = 0.1) -> str:
# #     def _replacer(match: re.Match) -> str:
# #         open_tag = match.group(1)
# #         bbox_str = match.group(2)
# #         close_tag = match.group(4)
# #         try:
# #             block_bbox = parse_bbox_from_string(bbox_str)
# #         except Exception as e:
# #             print(f"[WARN] Could not parse td bbox {bbox_str!r}: {e}")
# #             return f"{open_tag}{close_tag}"
# #         cell_words = get_words_in_block(block_bbox, words, overlap_threshold=overlap_threshold)
# #         return f"{open_tag}{' '.join(cell_words).strip()}{close_tag}"
# #     return _TD_WITH_BBOX_RE.sub(_replacer, html)

# # # ----------------------------
# # # Main pipeline
# # # ----------------------------
# # def main(
# #     bhaasha_root: str = "bhaashaocr_test",
# #     delta_torque_json_path: str = "delta_torque_raw.json",
# #     output_json_path: str = "delta_torque_with_bhaasha_ocr.json",
# #     overlap_threshold: float = 0.1
# # ) -> None:
# #     # Collect folders hin-1, hin-2, ...
# #     hin_folders = sorted([d for d in os.listdir(bhaasha_root)
# #                           if os.path.isdir(os.path.join(bhaasha_root, d)) and d.startswith("hin-")],
# #                          key=lambda x: int(x.split("-")[1]))

# #     words_by_folder: Dict[str, List[Tuple[Tuple[int, int, int, int], str]]] = {}
# #     for folder in hin_folders:
# #         folder_path = os.path.join(bhaasha_root, folder)
# #         try:
# #             words_by_folder[folder] = load_words_for_folder(folder_path)
# #         except Exception as e:
# #             print(f"[WARN] Skipping folder {folder}: {e}")

# #     # Load JSON
# #     with open(delta_torque_json_path, "r", encoding="utf-8") as f:
# #         data = json.load(f)
# #     rows = data if isinstance(data, list) else data.get("data", [])

# #     output_rows: List[Dict[str, Any]] = []
# #     for i, row in enumerate(rows):
# #         filename = row.get("filename", f"row-{i}")
# #         result_html = row.get("result", "")

# #         folder_name = f"hin-{i+1}"   # map by index
# #         words = words_by_folder.get(folder_name, [])

# #         if not words:
# #             print(f"[WARN] No words found for {folder_name} (filename {filename})")

# #         transformed_html = fill_td_cells_with_words(result_html, words, overlap_threshold)

# #         output_rows.append({
# #             "filename": filename,
# #             "result": row.get("result"),
# #             "bhaasha_ocr_html": transformed_html
# #         })

# #     with open(output_json_path, "w", encoding="utf-8") as f:
# #         json.dump(output_rows, f, ensure_ascii=False, indent=2)

# #     print(f"✅ Done. Wrote {len(output_rows)} records to {output_json_path}")


# # if __name__ == "__main__":
# #     main(
# #         bhaasha_root="bhaashaocr_test",   # <- adjust to your folder
# #         delta_torque_json_path="delta_torque_raw.json",
# #         output_json_path="delta_torque_with_bhaasha_ocr.json",
# #         overlap_threshold=0.00001,
# #     )
