import argparse
import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="SAHI + YOLOv11 检测脚本")
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像/视频路径或目录')
    parser.add_argument('--weights', type=str, required=True,
                        help='YOLOv11 权重文件路径（.pt）')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备')
    parser.add_argument('--slice_size', type=int, default=512,
                        help='切片大小')
    parser.add_argument('--overlap_ratio', type=float, default=0.2,
                        help='切片重叠率')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果保存目录')
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)



    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
        image_size=640,

        # fuse=True,
    )
    # ====================================================================


    if os.path.isdir(args.source):

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = [f for f in os.listdir(args.source)
                 if f.lower().endswith(image_extensions)]

        for file in files:
            image_path = os.path.join(args.source, file)
            print(f"处理图像: {image_path}")

            result = get_sliced_prediction(
                image=image_path,
                detection_model=detection_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
            )


            output_path = os.path.join(args.output_dir, f"sahi_{file}")
            result.export_visuals(export_dir=args.output_dir, file_name=f"sahi_{file}")
            print(f"结果已保存至: {output_path}")

    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):

        cap = cv2.VideoCapture(args.source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = os.path.join(args.output_dir, 'output_video.mp4')
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            print(f"处理帧: {frame_id}")


            result = get_sliced_prediction(
                image=frame[:, :, ::-1],  # BGR -> RGB
                detection_model=detection_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
            )


            from sahi.utils.cv import visualize_object_predictions
            visualized_frame = visualize_object_predictions(
                frame,
                object_prediction_list=result.object_prediction_list,
                rect_th=2,
                text_size=0.5,
                text_th=1,
            )
            writer.write(visualized_frame)
            frame_id += 1

        cap.release()
        writer.release()
        print(f"视频处理完成，保存至: {out_video}")

    else:

        print(f"处理图像: {args.source}")
        result = get_sliced_prediction(
            image=args.source,
            detection_model=detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
        )

        base_name = os.path.basename(args.source)
        output_path = os.path.join(args.output_dir, f"sahi_{base_name}")
        result.export_visuals(export_dir=args.output_dir, file_name=f"sahi_{base_name}")
        print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()