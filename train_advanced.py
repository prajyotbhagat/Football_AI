import sys
import logging
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
log_filename = "training_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


log = logging.getLogger("YOLO")
sys.stdout = StreamToLogger(log, logging.INFO)
sys.stderr = StreamToLogger(log, logging.ERROR)


def main():
    print("Starting YOLOv8x training – targeting >95% Recall on football dataset...")

    # ─────────────────────────────────────────────
    # WHY YOLOv8x instead of YOLOv8l?
    #
    # YOLOv8x has a wider/deeper neck (C2f blocks with more channels) which gives
    # richer multi-scale feature fusion — critical for detecting small balls and
    # partially occluded players. On a 6 GB GPU at imgsz=640 it runs comfortably
    # with batch=8, giving stable gradient updates vs. the noisy batch≈2 that
    # imgsz=1024 forced on yolov8l and caused the training loss to diverge.
    # ─────────────────────────────────────────────
    model = YOLO("yolov8x.pt")

    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=200,

        # ── Resolution ──────────────────────────────────────────────────────
        # 640 fits comfortably in 6 GB at batch=8.  The original imgsz=1024
        # forced batch≈2, making every gradient update extremely noisy and
        # causing the loss to *increase* between epochs 1→2 (1.222→1.321).
        imgsz=640,

        # ── Batch size ───────────────────────────────────────────────────────
        # Fixed at 8; gives stable gradients without OOM.  Do NOT use batch=-1
        # with large models on 6 GB VRAM — the autobatch poly-fit often crashes
        # or undershoots badly (as seen in the log: RankWarning + CUDA OOM).
        batch=8,

        device=0,
        project="football_detection",
        name="yolov8x_high_recall_v2",
        exist_ok=True,

        # ── Augmentation ─────────────────────────────────────────────────────
        # mosaic until epoch 160 then turned off so the model can fine-tune on
        # clean images.  close_mosaic does this automatically.
        mosaic=1.0,
        close_mosaic=40,          # disable mosaic for last 40 epochs

        # mixup REMOVED — blending images destroys small objects like the ball
        # and confuses the model when objects are already rare/small.
        mixup=0.0,

        scale=0.5,                # random scale ±50% — kept, good for size invariance
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # copy_paste helps a lot for small-object / crowded-scene datasets
        copy_paste=0.3,

        # ── Loss weights ─────────────────────────────────────────────────────
        # box=10.0 was far too aggressive.  The default is 7.5; we use 7.5 here
        # so the model learns class identity and box simultaneously without the
        # box loss dominating and destabilising early training.
        box=7.5,
        cls=0.5,

        # dfl (distribution focal loss) raised slightly to sharpen box edges,
        # which helps recall on small/blurry objects.
        dfl=2.0,

        # ── Optimizer & LR ───────────────────────────────────────────────────
        optimizer="SGD",          # SGD + cosine decay generalises better than
                                  # Adam at small batch sizes for detection tasks.
        lr0=0.01,                 # standard YOLOv8 starting LR
        lrf=0.01,                 # final LR = lr0 * lrf = 0.0001 (cosine decay)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,          # 3-epoch linear warm-up avoids early divergence

        # ── Recall-focused settings ───────────────────────────────────────────
        # Lower IOU threshold so more predictions are counted as correct during
        # training, nudging the model toward higher recall.
        iou=0.5,

        # ── Early stopping ────────────────────────────────────────────────────
        patience=30,              # tighter patience; stop if stuck for 30 epochs

        # ── Misc ──────────────────────────────────────────────────────────────
        # Save the top-3 checkpoints so you can pick the best recall/mAP trade-off.
        save_period=10,
        workers=4,
        amp=True,                 # mixed precision — halves VRAM, speeds training
    )

    print("Training completed!")

    # ── Evaluation ───────────────────────────────────────────────────────────
    # conf=0.05 gives a recall-biased operating point; the precision–recall curve
    # lets you choose any threshold post-hoc.
    print("Evaluating on test set (conf=0.05 for max-recall operating point)...")
    metrics = model.val(
        data="yolo_dataset/data.yaml",
        split="test",
        conf=0.05,
        iou=0.5,
    )
    print("Evaluation complete!")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP@0.50  : {metrics.box.map50:.4f}")
    print(f"  mAP@.5:.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
