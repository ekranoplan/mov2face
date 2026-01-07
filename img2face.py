import cv2
import os
from pathlib import Path
from insightface.app import FaceAnalysis

# dotenvを使用して.envファイルから設定を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

BASE_DIR = os.environ.get("KAO_BASE_DIR")
INPUT_DIR = Path(BASE_DIR + "/" + os.environ.get("KAO_SC_DIR"))
OUTPUT_DIR = Path(BASE_DIR + "/" + os.environ.get("KAO_FACE_DIR"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MARGIN = dict(left=0.45, right=0.45, top=0.45, bottom=0.45)

def blur_score_laplacian(bgr_img) -> float:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def make_square_bbox(x1, y1, x2, y2, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    nx1 = int(cx - side / 2)
    ny1 = int(cy - side / 2)
    nx2 = nx1 + side
    ny2 = ny1 + side

    # 画像境界でクリップ
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_w:
        shift = nx2 - img_w
        nx1 -= shift
        nx2 = img_w
    if ny2 > img_h:
        shift = ny2 - img_h
        ny1 -= shift
        ny2 = img_h

    # 最終安全クリップ
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return int(nx1), int(ny1), int(nx2), int(ny2)

def expand_bbox(x1, y1, x2, y2, img_w, img_h,
                left=0.25, right=0.25, top=0.45, bottom=0.25):
    """
    InsightFaceの顔bbox（顔面中心）を「頭部が余裕を持って収まる」ように拡張する。
    top（頭頂側）を多めに取るのがポイント。
    """
    w = x2 - x1
    h = y2 - y1
    nx1 = int(max(0, x1 - w * left))
    nx2 = int(min(img_w, x2 + w * right))
    ny1 = int(max(0, y1 - h * top))
    ny2 = int(min(img_h, y2 + h * bottom))
    return nx1, ny1, nx2, ny2

def main():
    # GPU優先（CUDA→CPUフォールバック）
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # ctx_id=0: GPU(0番) / det_sizeは小顔に強くしたければ上げる
    app.prepare(ctx_id=0, det_size=(640, 640))

    for img_path in INPUT_DIR.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel_path = img_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] load failed: {img_path}")
            continue

        faces = app.get(img)

        h, w = img.shape[:2]
        face_idx = 0
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)

            # 頭部が収まるように拡張
            x1, y1, x2, y2 = expand_bbox(
                x1, y1, x2, y2, w, h,
                **MARGIN
            )

            # 長辺基準で正方形にする
            x1, y1, x2, y2 = make_square_bbox(x1, y1, x2, y2, w, h)

            # crop
            head = img[y1:y2, x1:x2]
            if head.size == 0:
                continue

            if head.shape[0] < 256 or head.shape[1] < 256:
                continue

            score = blur_score_laplacian(head)
            print(f"[BLUR] {img_path.name} face{face_idx:02d} score={score:.1f}")
            if score < 120:
                continue

            TARGET_SIZE = 1024
            head = cv2.resize(head, (TARGET_SIZE, TARGET_SIZE),
                              interpolation=cv2.INTER_AREA)

            # 保存パス（複数顔対応）
            stem = img_path.stem
            suffix = img_path.suffix
            crop_path = out_path.with_name(f"{stem}_face{face_idx:02d}{suffix}")

            cv2.imwrite(str(crop_path), head)
            face_idx += 1
            print(f"[OK] {img_path} -> {out_path}  faces={len(faces)}")

if __name__ == "__main__":
    main()
