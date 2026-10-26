import cv2
import csv
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import torch


def count_objects(weights, source, conf_thres=0.25, classes=None, device='cpu',
                  save=False, view=False, csv_path=None):

    model = YOLO(weights)
    model.to(device)
    class_names = model.names

    source_path = Path(source)
    csv_file = None
    writer = None
    if csv_path:

        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_file)
    if source_path.is_dir():
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')) + list(
            source_path.glob('*.jpeg'))
        sorted_class_ids = sorted(class_names.keys())
        header = ['文件名'] + [class_names[i] for i in sorted_class_ids] + ['总计']
        if writer:
            writer.writerow(header)

        total_class_counts = defaultdict(int)
        total_images = 0
        total_objects = 0

        for img_path in image_files:
            results = model(img_path, conf=conf_thres, classes=classes, device=device)
            boxes = results[0].boxes
            if boxes is not None:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                class_counts = defaultdict(int)
                for cid in cls_ids:
                    class_counts[cid] += 1
                row = [img_path.name] + [class_counts.get(cid, 0) for cid in sorted_class_ids] + [len(boxes)]
            else:
                row = [img_path.name] + [0] * len(sorted_class_ids) + [0]

            if writer:
                writer.writerow(row)
            for cid, cnt in class_counts.items():
                total_class_counts[cid] += cnt
            total_objects += len(boxes) if boxes is not None else 0
            total_images += 1

            print(f"{img_path.name}: {len(boxes) if boxes else 0} objects")

            if save and boxes is not None:
                res_plotted = results[0].plot()
                save_dir = Path('runs/count')
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / img_path.name), res_plotted)
            if view and boxes is not None:
                results[0].show()
        if writer:
            summary_row = ['总计'] + [total_class_counts.get(cid, 0) for cid in sorted_class_ids] + [total_objects]
            writer.writerow(summary_row)
        print(f"\n总计图像数: {total_images}，总目标数: {total_objects}")

    elif source_path.suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(str(source_path))
        assert cap.isOpened(), "视频打开失败"

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save:
            save_video_path = Path('runs/count') / f"output_{source_path.stem}.mp4"
            save_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer_video = cv2.VideoWriter(str(save_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        sorted_class_ids = sorted(class_names.keys())
        header = (['帧号'] +
                  [f'帧_{class_names[i]}' for i in sorted_class_ids] +
                  ['帧_总计'] +
                  [f'累计_{class_names[i]}' for i in sorted_class_ids] +
                  ['累计_总计'])
        if writer:
            writer.writerow(header)

        frame_id = 0
        cumulative_ids_by_class = defaultdict(set)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.track(frame, conf=conf_thres, classes=classes, persist=True, device=device)
            boxes = results[0].boxes
            frame_class_counts = defaultdict(int)
            if boxes is not None and boxes.id is not None:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                for cid in cls_ids:
                    frame_class_counts[cid] += 1
                for cid, tid in zip(cls_ids, track_ids):
                    cumulative_ids_by_class[cid].add(tid)

            frame_total = len(boxes) if boxes is not None else 0
            cumulative_totals = {cid: len(cumulative_ids_by_class[cid]) for cid in sorted_class_ids}
            cumulative_total = sum(cumulative_totals.values())

            row = ([frame_id] +
                   [frame_class_counts.get(cid, 0) for cid in sorted_class_ids] +
                   [frame_total] +
                   [cumulative_totals.get(cid, 0) for cid in sorted_class_ids] +
                   [cumulative_total])
            if writer:
                writer.writerow(row)

            print(f"帧 {frame_id}: 当前帧目标数 {frame_total}, 累计独立目标数 {cumulative_total}")

            if save or view:
                annotated_frame = results[0].plot()
                if view:
                    cv2.imshow('YOLO Counting', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if save:
                    writer_video.write(annotated_frame)

        cap.release()
        if save:
            writer_video.release()
        cv2.destroyAllWindows()

        if writer:
            final_row = ['总计'] + [''] * len(sorted_class_ids) + [''] + [cumulative_totals.get(cid, 0) for cid in
                                                                        sorted_class_ids] + [cumulative_total]
            writer.writerow(final_row)
        print(f"\n视频处理完成，累计独立目标数: {cumulative_total}")

    else:

        results = model(source, conf=conf_thres, classes=classes, device=device)
        boxes = results[0].boxes

        sorted_class_ids = sorted(class_names.keys())
        header = ['文件名'] + [class_names[i] for i in sorted_class_ids] + ['总计']
        if writer:
            writer.writerow(header)

        if boxes is not None:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            class_counts = defaultdict(int)
            for cid in cls_ids:
                class_counts[cid] += 1
            row = [source_path.name] + [class_counts.get(cid, 0) for cid in sorted_class_ids] + [len(boxes)]
        else:
            row = [source_path.name] + [0] * len(sorted_class_ids) + [0]

        if writer:
            writer.writerow(row)

        print(f"图像 {source_path.name}: {len(boxes) if boxes else 0} objects")

        if save and boxes is not None:
            res_plotted = results[0].plot()
            save_dir = Path('runs/count')
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / source_path.name), res_plotted)
        if view and boxes is not None:
            results[0].show()

    if csv_file:
        csv_file.close()
        print(f"计数结果已保存至: {csv_path}")


if __name__ == "__main__":

    weights_path = "yolo26n.pt"
    data_source = "train2017"

    conf = 0.25
    classes_to_count = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_results = True
    show_window = False
    csv_output = "count_results.csv"

    count_objects(
        weights=weights_path,
        source=data_source,
        conf_thres=conf,
        classes=classes_to_count,
        device=device,
        save=save_results,
        view=show_window,
        csv_path=csv_output
    )
