import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO
import time
from datetime import datetime
import shutil
from sklearn.neighbors import KNeighborsRegressor



def _pjoin(base: str, *parts: str) -> str:
    cleaned = [base.rstrip("\\/")]
    cleaned.extend(p.strip("\\/") for p in parts if p is not None and str(p) != "")
    return "/".join(cleaned)



def remove_scattering_method1(data, Em, Ex, r1_mask, r2_mask):
    n_em, n_ex = data.shape
    result = data.copy()

    non_scatter_mask = ~(r1_mask | r2_mask)

    use_median_r1 = False
    median_ratio = 0.0

    if np.sum(r1_mask) > 0:
        result[r1_mask] = 0

        if np.sum(non_scatter_mask) >= 4:
            points = []
            values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_mask[i, j]:
                        points.append([Em[i], Ex[j]])
                        values.append(data[i, j])

            points = np.array(points)
            values = np.array(values)

            scatter_points = []
            scatter_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r1_mask[i, j]:
                        scatter_points.append([Em[i], Ex[j]])
                        scatter_indices.append((i, j))

            scatter_points = np.array(scatter_points)

            try:
                interpolator = CloughTocher2DInterpolator(points, values, fill_value=np.median(values))
                interpolated_values = interpolator(scatter_points)

                for idx_pred, (i, j) in enumerate(scatter_indices):
                    result[i, j] = interpolated_values[idx_pred]

                result[r1_mask] = np.maximum(result[r1_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r1_mask] = smoothed[r1_mask]
            except:
                result[r1_mask] = np.median(values) if len(values) > 0 else 0

    if np.sum(r2_mask) > 0:
        result[r2_mask] = 0

        non_scatter_for_r2 = ~r2_mask

        if np.sum(non_scatter_for_r2) >= 4:
            points = []
            values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_for_r2[i, j]:
                        points.append([Em[i], Ex[j]])
                        values.append(result[i, j])

            points = np.array(points)
            values = np.array(values)

            scatter_points = []
            scatter_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r2_mask[i, j]:
                        scatter_points.append([Em[i], Ex[j]])
                        scatter_indices.append((i, j))

            scatter_points = np.array(scatter_points)

            try:
                interpolator = CloughTocher2DInterpolator(points, values, fill_value=np.median(values))
                interpolated_values = interpolator(scatter_points)

                for idx_pred, (i, j) in enumerate(scatter_indices):
                    result[i, j] = interpolated_values[idx_pred]

                result[r2_mask] = np.maximum(result[r2_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r2_mask] = smoothed[r2_mask]
            except:
                result[r2_mask] = np.median(values) if len(values) > 0 else 0

    return result, use_median_r1, median_ratio


def remove_scattering_method2(data, Em, Ex, r1_mask, r2_mask, n_neighbors=10):
    n_em, n_ex = data.shape
    result = data.copy()

    non_scatter_mask = ~(r1_mask | r2_mask)

    use_median_r1 = False
    median_ratio = 0.0

    if np.sum(r1_mask) > 0:
        result[r1_mask] = 0

        if np.sum(non_scatter_mask) >= 4:
            points = []
            values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_mask[i, j]:
                        points.append([Em[i], Ex[j]])
                        values.append(data[i, j])

            points = np.array(points)
            values = np.array(values)

            scatter_points = []
            scatter_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r1_mask[i, j]:
                        scatter_points.append([Em[i], Ex[j]])
                        scatter_indices.append((i, j))

            scatter_points = np.array(scatter_points)

            try:
                interpolator = CloughTocher2DInterpolator(points, values, fill_value=np.median(values))
                interpolated_values = interpolator(scatter_points)

                for idx_pred, (i, j) in enumerate(scatter_indices):
                    result[i, j] = interpolated_values[idx_pred]

                result[r1_mask] = np.maximum(result[r1_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r1_mask] = smoothed[r1_mask]
            except:
                result[r1_mask] = np.median(values) if len(values) > 0 else 0

    if np.sum(r2_mask) > 0:
        result[r2_mask] = 0

        non_scatter_for_r2 = ~r2_mask

        if np.sum(non_scatter_for_r2) >= n_neighbors:
            train_features = []
            train_values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_for_r2[i, j]:
                        train_features.append([Em[i], Ex[j]])
                        train_values.append(result[i, j])

            train_features = np.array(train_features)
            train_values = np.array(train_values)

            predict_features = []
            predict_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r2_mask[i, j]:
                        predict_features.append([Em[i], Ex[j]])
                        predict_indices.append((i, j))

            predict_features = np.array(predict_features)

            try:
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
                knn_model.fit(train_features, train_values)
                predicted_values = knn_model.predict(predict_features)

                for idx_pred, (i, j) in enumerate(predict_indices):
                    result[i, j] = predicted_values[idx_pred]

                result[r2_mask] = np.maximum(result[r2_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r2_mask] = smoothed[r2_mask]
            except:
                result[r2_mask] = np.median(train_values) if len(train_values) > 0 else 0

    return result, use_median_r1, median_ratio


def remove_scattering_method3(data, Em, Ex, r1_mask, r2_mask, n_neighbors=10):
    n_em, n_ex = data.shape
    result = data.copy()

    non_scatter_mask = ~(r1_mask | r2_mask)

    use_median_r1 = False
    median_ratio = 0.0

    if np.sum(r1_mask) > 0:
        result[r1_mask] = 0

        if np.sum(non_scatter_mask) >= n_neighbors:
            train_features = []
            train_values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_mask[i, j]:
                        train_features.append([Em[i], Ex[j]])
                        train_values.append(data[i, j])

            train_features = np.array(train_features)
            train_values = np.array(train_values)

            predict_features = []
            predict_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r1_mask[i, j]:
                        predict_features.append([Em[i], Ex[j]])
                        predict_indices.append((i, j))

            predict_features = np.array(predict_features)

            try:
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
                knn_model.fit(train_features, train_values)
                predicted_values = knn_model.predict(predict_features)

                for idx_pred, (i, j) in enumerate(predict_indices):
                    result[i, j] = predicted_values[idx_pred]

                result[r1_mask] = np.maximum(result[r1_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r1_mask] = smoothed[r1_mask]
            except:
                result[r1_mask] = np.median(train_values) if len(train_values) > 0 else 0

    if np.sum(r2_mask) > 0:
        result[r2_mask] = 0

        non_scatter_for_r2 = ~r2_mask

        if np.sum(non_scatter_for_r2) >= 4:
            points = []
            values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_for_r2[i, j]:
                        points.append([Em[i], Ex[j]])
                        values.append(result[i, j])

            points = np.array(points)
            values = np.array(values)

            scatter_points = []
            scatter_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r2_mask[i, j]:
                        scatter_points.append([Em[i], Ex[j]])
                        scatter_indices.append((i, j))

            scatter_points = np.array(scatter_points)

            try:
                interpolator = CloughTocher2DInterpolator(points, values, fill_value=np.median(values))
                interpolated_values = interpolator(scatter_points)

                for idx_pred, (i, j) in enumerate(scatter_indices):
                    result[i, j] = interpolated_values[idx_pred]

                result[r2_mask] = np.maximum(result[r2_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r2_mask] = smoothed[r2_mask]
            except:
                result[r2_mask] = np.median(values) if len(values) > 0 else 0

    return result, use_median_r1, median_ratio


def remove_scattering_method4(data, Em, Ex, r1_mask, r2_mask, n_neighbors=10):
    n_em, n_ex = data.shape
    result = data.copy()

    non_scatter_mask = ~(r1_mask | r2_mask)

    use_median_r1 = False
    median_ratio = 0.0

    if np.sum(r1_mask) > 0:
        result[r1_mask] = 0

        if np.sum(non_scatter_mask) >= n_neighbors:
            train_features = []
            train_values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_mask[i, j]:
                        train_features.append([Em[i], Ex[j]])
                        train_values.append(data[i, j])

            train_features = np.array(train_features)
            train_values = np.array(train_values)

            predict_features = []
            predict_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r1_mask[i, j]:
                        predict_features.append([Em[i], Ex[j]])
                        predict_indices.append((i, j))

            predict_features = np.array(predict_features)

            try:
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
                knn_model.fit(train_features, train_values)
                predicted_values = knn_model.predict(predict_features)

                for idx_pred, (i, j) in enumerate(predict_indices):
                    result[i, j] = predicted_values[idx_pred]

                result[r1_mask] = np.maximum(result[r1_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r1_mask] = smoothed[r1_mask]
            except:
                result[r1_mask] = np.median(train_values) if len(train_values) > 0 else 0

    if np.sum(r2_mask) > 0:
        result[r2_mask] = 0

        non_scatter_for_r2 = ~r2_mask

        if np.sum(non_scatter_for_r2) >= n_neighbors:
            train_features = []
            train_values = []
            for i in range(n_em):
                for j in range(n_ex):
                    if non_scatter_for_r2[i, j]:
                        train_features.append([Em[i], Ex[j]])
                        train_values.append(result[i, j])

            train_features = np.array(train_features)
            train_values = np.array(train_values)

            predict_features = []
            predict_indices = []
            for i in range(n_em):
                for j in range(n_ex):
                    if r2_mask[i, j]:
                        predict_features.append([Em[i], Ex[j]])
                        predict_indices.append((i, j))

            predict_features = np.array(predict_features)

            try:
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
                knn_model.fit(train_features, train_values)
                predicted_values = knn_model.predict(predict_features)

                for idx_pred, (i, j) in enumerate(predict_indices):
                    result[i, j] = predicted_values[idx_pred]

                result[r2_mask] = np.maximum(result[r2_mask], 0)
                smoothed = gaussian_filter(result, sigma=0.5)
                result[r2_mask] = smoothed[r2_mask]
            except:
                result[r2_mask] = np.median(train_values) if len(train_values) > 0 else 0

    return result, use_median_r1, median_ratio



def detect_scattering_region(model, img_path, Em, Ex, r1_width, r2_width,
                              conf_threshold=0.5, device="cpu"):
    img = cv2.imread(img_path)
    if img is None:
        return False, False, False, 0, 0

    detect_start = time.time()
    results = model(img, conf=conf_threshold, device=device, verbose=False)
    detections = results[0].boxes
    detect_time = time.time() - detect_start

    if len(detections) == 0:
        return False, False, False, detect_time, 0

    img_height, img_width = img.shape[:2]

    in_r1 = False
    in_r2 = False

    plot_left = img_width * 0.12
    plot_right = img_width * 0.82
    plot_top = img_height * 0.08
    plot_bottom = img_height * 0.88

    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    em_min, em_max = Em.min(), Em.max()
    ex_min, ex_max = Ex.min(), Ex.max()

    for box in detections:
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        em_center = em_min + (center_x - plot_left) / plot_width * (em_max - em_min)
        ex_center = ex_max - (center_y - plot_top) / plot_height * (ex_max - ex_min)

        if abs(em_center - ex_center) <= r1_width * 2:
            in_r1 = True

        if abs(em_center - 2 * ex_center) <= r2_width * 2:
            in_r2 = True

    return True, in_r1, in_r2, detect_time, len(detections)



def run_single_method(method_id, input_folder, output_folder, picture_folder,
                      model, r1_width, r2_width,
                      conf_threshold=0.5, n_neighbors=10,
                      direct_output=False,
                      max_files=None):
    method_names = {
        4: "First-order KNN + Second-order KNN"
    }

    method_funcs = {
        4: remove_scattering_method4
    }

    print(f"\n{'='*70}")
    print(f"Method {method_id}: {method_names[method_id]}")
    print(f"{'='*70}")
    print(f"First-order scatter width: {r1_width} nm")
    print(f"Second-order scatter width: {r2_width} nm")

    if direct_output:
        method_output = output_folder
        method_picture = picture_folder
        method_detection = _pjoin(picture_folder, "YOLO_detection")
    else:
        method_output = _pjoin(output_folder, f"method_{method_id}_data")
        method_picture = _pjoin(picture_folder, f"method_{method_id}_images")
        method_detection = _pjoin(picture_folder, f"method_{method_id}_yolo_detection")
    os.makedirs(method_output, exist_ok=True)
    os.makedirs(method_picture, exist_ok=True)
    os.makedirs(method_detection, exist_ok=True)

    excel_files = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.endswith('.xlsx') or f.endswith('.xls'):
                full_path = _pjoin(root, f)
                rel_path = os.path.relpath(full_path, input_folder)
                excel_files.append(rel_path)
    excel_files = sorted(excel_files)

    if max_files is not None:
        try:
            max_files_int = int(max_files)
        except Exception:
            max_files_int = None
        if max_files_int is not None and max_files_int > 0:
            excel_files = excel_files[:max_files_int]

    if not excel_files:
        print("No Excel files found.")
        method_summary = {
            'method_id': method_id,
            'method_name': method_names[method_id],
            'total_time': 0,
            'file_count': 0,
            'completed_count': 0,
            'avg_iterations': 0,
            'avg_read_time': 0,
            'avg_process_time': 0,
            'avg_plot_time': 0,
            'avg_yolo_time': 0
        }
        return [], [], method_summary

    print(f"Files to process: {len(excel_files)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    process_stats = []
    method_start_time = time.time()

    print(f"\n--- Single-pass processing ---")
    print(f"Files queued: {len(excel_files)}")

    for fname in excel_files:
        file_start_time = time.time()
        file_path = _pjoin(input_folder, fname)
        file_basename = os.path.basename(fname)
        base_name = os.path.splitext(file_basename)[0]

        try:
            read_start = time.time()
            df = pd.read_excel(file_path, header=None)
            Em = np.array(df.iloc[1:, 0], dtype=float)
            Ex = np.array(df.iloc[0, 1:], dtype=float)
            data = df.iloc[1:, 1:].values.astype(float)
            read_time = time.time() - read_start

            n_em, n_ex = data.shape

            r1_mask = np.zeros((n_em, n_ex), dtype=bool)
            r2_mask = np.zeros((n_em, n_ex), dtype=bool)

            for i in range(n_em):
                for j in range(n_ex):
                    if abs(Em[i] - Ex[j]) <= r1_width:
                        r1_mask[i, j] = True
                    elif abs(Em[i] - 2 * Ex[j]) <= r2_width:
                        r2_mask[i, j] = True

            process_start = time.time()
            result, _, _ = method_funcs[method_id](data, Em, Ex, r1_mask, r2_mask, n_neighbors)
            process_time = time.time() - process_start

            df.iloc[1:, 1:] = result
            output_file = _pjoin(method_output, file_basename)
            df.to_excel(output_file, index=False, header=False)

            plot_start = time.time()
            pic_path = _pjoin(method_picture, f"{base_name}.png")

            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(6, 4.8))
            cs = ax.contourf(Em, Ex, result.T, levels=100, cmap="jet")
            cbar = plt.colorbar(cs, ax=ax)
            cbar.set_label("Fluorescence intensity (a.u.)")
            ax.set_xlabel("Emission (nm)")
            ax.set_ylabel("Excitation (nm)")
            ax.set_title(f"{base_name}")
            plt.tight_layout()
            plt.savefig(pic_path, dpi=300)
            plt.close(fig)
            plot_time = time.time() - plot_start

            yolo_start = time.time()
            has_scatter, in_r1, in_r2, _, num_det = detect_scattering_region(
                model, pic_path, Em, Ex, r1_width, r2_width, conf_threshold, device
            )
            yolo_time = time.time() - yolo_start

            if has_scatter:
                img = cv2.imread(pic_path)
                results_yolo = model(img, conf=conf_threshold, device=device, verbose=False)
                annotated = results_yolo[0].plot()
                detection_path = _pjoin(method_detection, f"{base_name}.png")
                cv2.imwrite(detection_path, annotated)

            total_file_time = time.time() - file_start_time

            process_stats.append({
                'file_name': fname,
                'method': f"method_{method_id}",
                'pass_index': 1,
                'first_order_width_nm': r1_width,
                'second_order_width_nm': r2_width,
                'read_time_s': round(read_time, 4),
                'scatter_process_time_s': round(process_time, 4),
                'plot_time_s': round(plot_time, 4),
                'yolo_time_s': round(yolo_time, 4),
                'total_file_time_s': round(total_file_time, 4),
                'first_order_points': int(np.sum(r1_mask)),
                'second_order_points': int(np.sum(r2_mask)),
                'yolo_detection_count': num_det,
                'detected_first_order': 'yes' if in_r1 else 'no',
                'detected_second_order': 'yes' if in_r2 else 'no',
                'completed': 'yes'
            })

            print(f"  {fname}: completed | r1={r1_width}nm, r2={r2_width}nm | detections:{num_det}")

        except Exception as e:
            print(f"  {fname}: failed - {str(e)}")
            process_stats.append({
                'file_name': fname,
                'method': f"method_{method_id}",
                'pass_index': 1,
                'first_order_width_nm': r1_width,
                'second_order_width_nm': r2_width,
                'read_time_s': 0,
                'scatter_process_time_s': 0,
                'plot_time_s': 0,
                'yolo_time_s': 0,
                'total_file_time_s': 0,
                'first_order_points': 0,
                'second_order_points': 0,
                'yolo_detection_count': 0,
                'detected_first_order': 'error',
                'detected_second_order': 'error',
                'completed': 'error'
            })

    method_total_time = time.time() - method_start_time

    file_summary_stats = []
    for stat in process_stats:
        file_summary_stats.append({
            'file_name': stat['file_name'],
            'method': stat['method'],
            'pass_index': stat['pass_index'],
            'first_order_width_nm': stat['first_order_width_nm'],
            'second_order_width_nm': stat['second_order_width_nm'],
            'total_read_time_s': stat['read_time_s'],
            'total_process_time_s': stat['scatter_process_time_s'],
            'total_plot_time_s': stat['plot_time_s'],
            'total_yolo_time_s': stat['yolo_time_s'],
            'total_file_time_s': stat['total_file_time_s'],
            'completed': 'yes' if stat['completed'] == 'yes' else 'no'
        })

    completed_count = sum(1 for s in file_summary_stats if s['completed'] == 'yes')
    avg_iterations = 1.0

    print(f"\n--- Method {method_id} summary ---")
    print(f"Processed files: {len(excel_files)}")
    print(f"Completed files: {completed_count}/{len(excel_files)}")
    print(f"Average passes: {avg_iterations:.2f}")
    print(f"Method total time: {method_total_time:.2f} s")

    if file_summary_stats:
        avg_read = np.mean([s['total_read_time_s'] for s in file_summary_stats])
        avg_process = np.mean([s['total_process_time_s'] for s in file_summary_stats])
        avg_plot = np.mean([s['total_plot_time_s'] for s in file_summary_stats])
        avg_yolo = np.mean([s['total_yolo_time_s'] for s in file_summary_stats])
        print(f"Average read time: {avg_read:.4f} s")
        print(f"Average process time: {avg_process:.4f} s")
        print(f"Average plot time: {avg_plot:.4f} s")
        print(f"Average YOLO time: {avg_yolo:.4f} s")

    method_summary = {
        'method_id': method_id,
        'method_name': method_names[method_id],
        'total_time': method_total_time,
        'file_count': len(excel_files),
        'completed_count': completed_count,
        'avg_iterations': avg_iterations,
        'avg_read_time': np.mean([s['total_read_time_s'] for s in file_summary_stats]) if file_summary_stats else 0,
        'avg_process_time': np.mean([s['total_process_time_s'] for s in file_summary_stats]) if file_summary_stats else 0,
        'avg_plot_time': np.mean([s['total_plot_time_s'] for s in file_summary_stats]) if file_summary_stats else 0,
        'avg_yolo_time': np.mean([s['total_yolo_time_s'] for s in file_summary_stats]) if file_summary_stats else 0
    }

    return process_stats, file_summary_stats, method_summary



def run_all_methods_independently(input_folder, output_folder, picture_folder,
                                   model_path, start_r1_width=25, start_r2_width=15,
                                   conf_threshold=0.5, n_neighbors=10,
                                   methods_to_run=None,
                                   output_folders_by_method=None,
                                   picture_folders_by_method=None,
                                   max_files=None):
    methods_to_run = [4]

    print("\n" + "=" * 80)
    print("KNN_KNN single-pass Rayleigh scattering removal")
    print("=" * 80)
    print(f"Method 4: first-order KNN fitting + second-order KNN fitting")
    print("-" * 80)
    print(f"Methods to run: {methods_to_run}")
    print(f"Initial first-order width: {start_r1_width} nm")
    print(f"Initial second-order width: {start_r2_width} nm")
    print(f"Run mode: single pass (no iteration)")
    print(f"YOLO confidence threshold: {conf_threshold}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(picture_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    model = YOLO(model_path)
    print(f"YOLO model loaded: {model_path}\n")

    total_start_time = time.time()

    all_process_stats = []
    all_file_summaries = []
    all_method_summaries = []

    for method_id in methods_to_run:
        method_output_folder = output_folder
        method_picture_folder = picture_folder
        direct_output = False
        if output_folders_by_method and method_id in output_folders_by_method:
            method_output_folder = output_folders_by_method[method_id]
            direct_output = True
        if picture_folders_by_method and method_id in picture_folders_by_method:
            method_picture_folder = picture_folders_by_method[method_id]
            direct_output = True

        process_stats, file_summaries, method_summary = run_single_method(
            method_id=method_id,
            input_folder=input_folder,
            output_folder=method_output_folder,
            picture_folder=method_picture_folder,
            model=model,
            r1_width=start_r1_width,
            r2_width=start_r2_width,
            conf_threshold=conf_threshold,
            n_neighbors=n_neighbors,
            direct_output=direct_output,
            max_files=max_files
        )
        all_process_stats.extend(process_stats)
        all_file_summaries.extend(file_summaries)
        all_method_summaries.append(method_summary)

    total_time = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("")
    print("=" * 80)
    print(f": {total_time:.2f}  ({total_time/60:.2f} )")
    print(f": ")
    print(f"\n:")
    print("-" * 70)
    print(f"{'':<35} {'()':<12} {'':<10} {''}")
    print("-" * 70)
    for ms in all_method_summaries:
        print(f"{ms['method_id']}: {ms['method_name']:<25} {ms['total_time']:<12.2f} {ms['avg_iterations']:<10.2f} {ms['completed_count']}/{ms['file_count']}")
    print("-" * 70)

    print(f"\n: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'process_stats': all_process_stats,
        'file_summaries': all_file_summaries,
        'method_summaries': all_method_summaries,
        'total_time': total_time
    }



if __name__ == "__main__":
    input_folder = "../data/raw"
    output_folder = "../data/process/eem"
    picture_folder = "../picture"

    output_folders_by_method = {4: "../data/process/eem"}
    picture_folders_by_method = {4: "../picture/KNN_KNN"}
    model_path = "../docs/best.pt"


    start_r1_width = 15

    start_r2_width = 40
    conf_threshold = 0.95
    n_neighbors = 10
    max_files = None

    methods_to_run = [4]

    results = run_all_methods_independently(
        input_folder=input_folder,
        output_folder=output_folder,
        picture_folder=picture_folder,
        model_path=model_path,
        start_r1_width=start_r1_width,
        start_r2_width=start_r2_width,
        conf_threshold=conf_threshold,
        n_neighbors=n_neighbors,
        methods_to_run=methods_to_run,
        output_folders_by_method=output_folders_by_method,
        picture_folders_by_method=picture_folders_by_method,
        max_files=max_files
    )

    print("\n" + "=" * 80)
    print("")
    print("=" * 80)
