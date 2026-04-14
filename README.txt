AIRBORNE DETECTION - README
===========================

Overview
--------
This codebase is a multi-model pipeline for detecting and tracking airborne objects (drones,
birds, aircraft) in aerial video footage. It was developed for the AIcrowd Airborne Object
Tracking Challenge. Objects are extremely small (on average 0.01% of the image), so the system
uses a two-stage approach: background alignment (to compensate for camera motion) followed by
segmentation-based detection and multi-object tracking.

Key components:
  - seg_tracker/      Main detection and tracking module (HRNet/DLA/EfficientDet-based)
  - core/             Dataset loading, flight/frame/object data structures
  - evaluator/        Submission interface (AirbornePredictor base class)
  - utility/          Visualization tools
  - scripts/          SLURM job submission scripts


Directory Structure
-------------------
airborne_detection/
  core/
    dataset.py                    Dataset and S3 download
    flight.py / frame.py          Flight and frame data structures
    airborne_object.py            Object class
    airborne_object_location.py   Bounding box and metadata
    metrics/                      Evaluation metrics
  seg_tracker/
    train.py                      Train segmentation model
    train_transformation.py       Train background alignment model
    inference.py                  Run inference / evaluation
    seg_tracker.py                SegDetector and SegTrackerFromOffset classes
    tracking.py                   SimpleOffsetTracker (multi-object tracking)
    dataset_tracking.py           Training dataset
    dataset_transform.py          Transformation training dataset
    models_segmentation.py        HRNet, DLA, EfficientDet architectures
    models_transformation.py      Background alignment architecture
    common_utils.py               Shared utilities
    config.py                     Runtime constants
    experiments/                  YAML config files per experiment
  evaluator/
    airborne_detection.py         AirbornePredictor base class
  utility/
    vis_writer.py                 Video visualization writer
  scripts/
    RGB_tracking_training.sh      SLURM job: train segmentation model
    run_predictions.sh            SLURM job: precompute transformation offsets
    RGB_tracking.sh               SLURM job: evaluate RGB tracking
    Evaluation.sh                 SLURM job: full evaluation pipeline
  test.py                         Random predictor baseline
  seg_test.py                     Segmentation tracker submission entry point
  run_my_video.py                 Process a custom video with tracking
  debug.py                        Dataset exploration utility


Environment Setup
-----------------
Python 3.9.18 is used via the module system. A virtual environment is in:
  $HOME/airborne_detection/airbone/

To activate it:
  module load stack/.2024-05-silent gcc/13.2.0 python/3.9.18
  module load eth_proxy
  source $HOME/airborne_detection/airbone/bin/activate

To install dependencies from scratch:
  pip install -r requirements.txt
  pip install -r seg_tracker/requirements.txt

System packages needed (see apt.txt):
  build-essential, libpython3.7-dev, libxrender1, libsm6, libglib2.0-0, libxext6


Training
--------
There are two separate training stages:

Stage 1 - Train the background alignment (transformation) model
  This model estimates the frame-to-frame camera transformation (optical flow / homography)
  so that background motion can be subtracted before object detection.

  a) Train the model:
       cd seg_tracker
       python -u train_transformation.py train 030_tr_tsn_rn34_w3_crop_borders

  b) Precompute transformation offsets for the full dataset (run as SLURM job):
       sbatch scripts/run_predictions.sh
     This runs:
       python -u train_transformation.py predict_dataset_offsets \
           030_tr_tsn_rn34_w3_crop_borders --part part1
     ...for each dataset part and saves results to /cluster/scratch/USERNAME/.

Stage 2 - Train the segmentation / detection model
  This model detects small airborne objects from pairs of consecutive (motion-compensated) frames.

  Directly (interactive or on a GPU node):
       cd seg_tracker
       python -u train.py train 120_hrnet32_all

  As a SLURM job (recommended for full training, 48 hours):
       sbatch scripts/RGB_tracking_training.sh

  The training script copies the dataset tarball to the local SSD ($TMPDIR) for fast I/O:
       TAR_FILE=/cluster/scratch/USERNAME/airborne_dataset_new_with_transforms.tar
     After extraction, it sets FAST_DATA_DIR and starts training.

  Experiment config files are YAML files in seg_tracker/experiments/:
    120_hrnet32_all.yaml           HRNet-W32 (default, best single model)
    120_dla60_256_sgd_all_rerun.yaml  DLA-60 variant
    120_edet_b5_all.yaml           EfficientDet-B5 variant
    120_gernet_m_b2_all.yaml       GenEfficientNet variant
    030_tr_tsn_rn34_w3_crop_borders.yaml  Transformation model (ResNet-34)

  Key training hyperparameters (from 120_hrnet32_all.yaml as example):
    model_cls:     HRNetSegmentation
    base_model:    hrnet_w32
    input_frames:  2 (consecutive frames as input)
    batch_size:    16
    initial_lr:    2.5e-5
    optimizer:     MADGRAD
    scheduler:     CosineAnnealingWarmRestarts
    nb_epochs:     4000
    loss weights:  cls=10000, size=1, offset=0.25, tracking=1, above_horizon=0.1

  Outputs are saved to:
    output/checkpoints/   Saved model checkpoints
    output/tensorboard/   TensorBoard logs (view with: tensorboard --logdir output/tensorboard)
    output/oof/           Out-of-fold predictions


Loading Data
------------
The dataset follows the AIcrowd Airborne Object Tracking format.

Dataset location on the cluster:
  Permanent storage:  /cluster/scratch/USERNAME/airborne_dataset_new/
  Tarball (for jobs):  /cluster/scratch/USERNAME/airborne_dataset_new_with_transforms.tar
  Original challenge data (S3): s3://airborne-obj-detection-challenge-training/part1|part2|part3

Loading via the Dataset class (core/dataset.py):

  from core.dataset import Dataset

  dataset = Dataset(
      local_path='/cluster/scratch/USERNAME/airborne_dataset_new',
      s3_path='s3://airborne-obj-detection-challenge-training/part1/',
      download_if_required=True,
      partial=False    # True to use the smaller 500GB subset
  )

  flights = dataset.get_flight_ids()
  flight   = dataset.get_flight('some_flight_id')

  for frame_id in flight.frames:
      frame = flight.get_frame(frame_id)
      for obj_id, loc in frame.detected_object_locations.items():
          # loc.bb: bounding box with .left, .top, .width, .height
          # loc.distance: distance in metres (only for "planned" objects)
          # loc.is_above_horizon: 1=above, -1=below, 0=unclear
          print(loc.bb.left, loc.bb.top, loc.bb.width, loc.bb.height)

Ground truth JSON format (one file per flight):
  {
    "flight_id": "...",
    "frames": {
      "<timestamp><flight_id>": {
        "entities": [
          {
            "id": "...",
            "bb": [left, top, width, height],   // pixel coordinates, center-based
            "range_distance_m": 150.0,
            "is_above_horizon": 1
          }
        ]
      }
    }
  }

Image properties:
  Resolution: 2448 x 2048 pixels
  Format: PNG (grayscale for IR, RGB for optical)
  Naming: {timestamp}{flight_id}.png

Dataset exploration:
  python debug.py   # Prints dataset statistics and sample detections


Testing on New Data
-------------------
There are three ways to run inference:

1. Using the inference script (seg_tracker/inference.py)
   This script supports multiple modes. Run from the seg_tracker/ directory.

   RGB segmentation evaluation:
     python inference.py --mode evaluate_rgb_baseline \
         --video /path/to/video.mp4 \
         --exp 120_hrnet32_all

   Fused RGB+IR evaluation:
     python inference.py --mode evaluate_fused \
         --video /path/to/rgb_video.mp4 \
         --ir_video /path/to/ir_video.mp4 \
         --fused_weights /path/to/fused_weights.pth \
         --conf_threshold 0.15

   IR-only evaluation:
     python inference.py --mode evaluate_ir \
         --ir_video /path/to/ir_video.mp4 \
         --exp 120_hrnet32_all

   Visualization (saves annotated video):
     python inference.py --mode visualize \
         --video /path/to/video.mp4 \
         --exp 120_hrnet32_all

   YOLO-based inference modes (alternative):
     python inference.py --mode yolo_rgb --yolo_weights path/to/rgb_best.pt --video video.mp4
     python inference.py --mode yolo_ir  --yolo_weights path/to/ir_best.pt  --ir_video video.mp4
     python inference.py --mode yolo_rgbt \
         --yolo_weights weights.pt --video rgb.mp4 --ir_video ir.mp4

   As a SLURM job:
     sbatch scripts/RGB_tracking.sh        # RGB baseline
     sbatch scripts/Evaluation.sh          # Full evaluation pipeline

2. Using the submission entry point (seg_test.py)
   This is the AIcrowd submission format. It processes all flights in the dataset.

     python seg_test.py

   Internally this uses SegPredictor (subclass of AirbornePredictor) which:
     - Loads SegDetector (transformation model + segmentation model)
     - Wraps it in SegTrackerFromOffset for multi-frame tracking
     - Iterates over frames in each flight, registers detections

3. Processing a custom video (run_my_video.py)
   For running on arbitrary video files outside of the challenge dataset format:

     python run_my_video.py --video /path/to/video.mp4

Implementing your own predictor:
  Subclass AirbornePredictor from evaluator/airborne_detection.py:

    from evaluator.airborne_detection import AirbornePredictor

    class MyPredictor(AirbornePredictor):
        def inference_setup(self):
            # Load your model here
            pass

        def inference(self, flight_id):
            self.flight_started()
            for frame_image in self.get_all_frame_images(flight_id):
                img_path = self.get_frame_image_location(flight_id, frame_image)
                frame = cv2.imread(img_path)
                # Run detection...
                for detection in results:
                    cx, cy, w, h = detection
                    bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]  # [x0,y0,x1,y1]
                    self.register_object_and_location(
                        'airborne',    # class name
                        track_id,      # integer track ID
                        bbox,          # [x0, y0, x1, y1] in pixels
                        confidence,    # float 0 < conf < 1
                        frame_image    # filename string
                    )

Using the Detector and Tracker Directly
  from seg_tracker.seg_tracker import SegDetector, SegTrackerFromOffset

  detector = SegDetector(exp_name='120_hrnet32_all')
  tracker  = SegTrackerFromOffset(detector=detector)

  prev_frame = None
  for frame in frames:
      if prev_frame is None:
          prev_frame = frame
          continue
      results = tracker.predict(image=frame, prev_image=prev_frame)
      prev_frame = frame
      # results: list of dicts with keys: track_id, conf, cx, cy, w, h


Utilities
---------

common_utils.py (seg_tracker/common_utils.py)
  - load_config_data(exp_name)           Load YAML config for an experiment
  - normalize_experiment_name(name)      Strip path/extension from config name
  - build_geom_transform(points, ...)    Build affine/homography transform matrix
  - argmax2d(arr) / argmin2d(arr)        Find 2D index of max/min in an array
  - AverageMeter                         Track running average of a metric
  - DotDict                              Dict with attribute access (config['key'] -> config.key)
  - timeit_context(label)                Context manager to print elapsed time

  Example:
    from seg_tracker.common_utils import load_config_data, AverageMeter, timeit_context

    config = load_config_data('120_hrnet32_all')  # loads experiments/120_hrnet32_all.yaml

    meter = AverageMeter()
    meter.update(loss.item())
    print(meter.avg)

    with timeit_context('forward pass'):
        output = model(input)   # prints "forward pass: 12.3ms"

vis_writer.py (utility/vis_writer.py)
  - VisWriter(output_path, fps=10, mode='RGB')   Write annotated frames to video
  - writer.write_frame(frame)                     Append a frame
  - writer.close()                                Finalize (runs ffmpeg compression)

  Example:
    from utility.vis_writer import VisWriter
    writer = VisWriter('/tmp/output.mp4', fps=10, mode='GRAY')
    for frame in frames:
        writer.write_frame(annotated_frame)
    writer.close()

tracking.py (seg_tracker/tracking.py)
  - SimpleOffsetTracker                  Multi-object tracker
    - __init__(conf_threshold, dist_threshold)
    - update(detections, frame_offset)   Returns detections with stable track IDs
    - Track history kept for up to 3 frames; uses velocity-based matching

seg_prediction_to_items.py (seg_tracker/seg_prediction_to_items.py)
  - predictions_to_detection_items(pred, conf_threshold)
    Converts raw segmentation model output to a list of DetectionItem objects
    Each DetectionItem has: cx, cy, w, h, confidence, distance

predict_ensemble.py (seg_tracker/predict_ensemble.py)
  - Combine predictions from multiple models
  - Weighted averaging of confidence scores

file_handler.py (core/file_handler.py)
  - FileHandler(local_path, s3_path)    Access files from S3 with local caching
  - handler.get_file(path)              Download and cache file locally
  - Supports parallel multi-file downloads


Configuration Constants (seg_tracker/config.py)
  DATA_DIR            Base data directory
  OFFSET_SCALE        256.0 (scale factor for flow offsets)
  UPPER_BOUND_MIN_DIST  330  (min detection range in metres)
  UPPER_BOUND_MAX_DIST  700  (max detection range in metres)
  MIN_OBJECT_AREA     100    (minimum object area in pixels^2)
  MIN_SECS            3.0    (minimum track duration in seconds)
  CLASSES             ['None','Airborne','Airplane','Bird','Drone','Flock','Helicopter']


Notes and Tips
--------------
- Dataset I/O is the bottleneck: the training scripts copy the full dataset tar
  to the local SSD ($TMPDIR, 700GB) before training to avoid slow network reads.
- The .pkl cache files in the dataset directory (*.pkl) can become stale if the
  data changes. Delete them if you see unexpected dataset sizes:
    rm -f $FAST_DATA_DIR/*.pkl
- TensorBoard logs are written to output/tensorboard/. To monitor training:
    tensorboard --logdir output/tensorboard --port 6006
- The transformation model (Stage 1) only needs to be trained once; its precomputed
  offsets are stored alongside the dataset and reused during segmentation training.
- Multiple model architectures (HRNet, DLA, EfficientDet) can be ensembled at
  inference time using predict_ensemble.py for improved accuracy.
- The FAST_DATA_DIR environment variable, if set, overrides the default data path
  in the training scripts (used by the SLURM jobs for SSD-local data).
