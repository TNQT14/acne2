"""
Script đánh giá mô hình YOLOv8 multiclass cho phát hiện mụn
Tạo confusion matrix và các metrics đánh giá
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import torch
import cv2

# Cấu hình
MODEL_PATH = "/Users/quangthai/Documents/AI in Bioinfomatics/acne2/report/multi-class-acne-detection/acne_yolov8_multiclass/weights/best.pt"
DATA_YAML = "/Users/quangthai/Documents/AI in Bioinfomatics/acne2/data-2/data.yaml"
TEST_IMAGES_DIR = "/Users/quangthai/Documents/AI in Bioinfomatics/acne2/data-2/test/images"
OUTPUT_DIR = "/Users/quangthai/Documents/AI in Bioinfomatics/acne2/report/multi-class-acne-detection/acne_yolov8_multiclass"

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set matplotlib backend để tránh lỗi khi chạy trên server
plt.switch_backend('Agg')

def load_data_config(data_yaml_path):
    """Load cấu hình dataset từ data.yaml"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config

def load_ground_truth_labels(label_file):
    """Load ground truth labels từ file .txt (YOLO format)"""
    labels = []
    if not os.path.exists(label_file):
        return labels
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                labels.append(class_id)
    return labels

def evaluate_model_on_test(model_path, data_yaml_path, test_images_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Đánh giá mô hình trên test set và thu thập predictions với ground truth
    """
    # Load model
    print(f"Đang load mô hình từ: {model_path}")
    model = YOLO(model_path)
    
    # Load data config
    data_config = load_data_config(data_yaml_path)
    class_names = data_config['names']
    num_classes = len(class_names)
    
    print(f"Số lớp: {num_classes}")
    print(f"Tên các lớp: {class_names}")
    
    # Lấy danh sách ảnh test
    test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    print(f"Số ảnh test: {len(test_images)}")
    
    # Thu thập predictions và ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Dictionary để lưu predictions và ground truth theo từng class
    predictions_by_class = {i: [] for i in range(num_classes)}
    ground_truth_by_class = {i: [] for i in range(num_classes)}
    
    print("\nĐang đánh giá mô hình trên test set...")
    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 10 == 0:
            print(f"Đã xử lý: {idx + 1}/{len(test_images)} ảnh")
        
        # Lấy label file tương ứng
        label_file = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        
        # Load ground truth
        gt_labels = load_ground_truth_labels(label_file)
        
        # Dự đoán
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Lấy predictions
        pred_labels = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
            pred_conf = results[0].boxes.conf.cpu().numpy()
            pred_labels = pred_classes.tolist()
        
        # Thêm vào danh sách tổng
        all_ground_truth.extend(gt_labels)
        all_predictions.extend(pred_labels)
        
        # Thêm vào dictionary theo class
        for gt_label in gt_labels:
            ground_truth_by_class[gt_label].append(gt_label)
        
        for pred_label in pred_labels:
            predictions_by_class[pred_label].append(pred_label)
    
    print(f"\nTổng số objects trong ground truth: {len(all_ground_truth)}")
    print(f"Tổng số objects được phát hiện: {len(all_predictions)}")
    
    # Thống kê theo class
    print("\n=== Thống kê Ground Truth theo class ===")
    for i, class_name in enumerate(class_names):
        count = len(ground_truth_by_class[i])
        print(f"{class_name}: {count}")
    
    print("\n=== Thống kê Predictions theo class ===")
    for i, class_name in enumerate(class_names):
        count = len(predictions_by_class[i])
        print(f"{class_name}: {count}")
    
    return all_ground_truth, all_predictions, class_names, num_classes

def create_confusion_matrix(ground_truth, predictions, class_names, output_dir):
    """
    Tạo và vẽ confusion matrix
    """
    print("\nĐang tạo confusion matrix...")
    
    # Tạo confusion matrix
    # Thêm tất cả các class có thể (ngay cả khi không có trong data)
    num_classes = len(class_names)
    cm = confusion_matrix(
        ground_truth, 
        predictions, 
        labels=list(range(num_classes))
    )
    
    # Normalize confusion matrix (tính theo phần trăm)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_percent = cm_normalized * 100
    
    # Tạo figure với 2 subplots: absolute và normalized
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Confusion matrix với số lượng tuyệt đối
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Số lượng'}
    )
    axes[0].set_xlabel('Dự đoán (Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Thực tế (Actual)', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix - Số lượng tuyệt đối', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 2: Confusion matrix với phần trăm (normalized)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Phần trăm (%)'}
    )
    axes[1].set_xlabel('Dự đoán (Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Thực tế (Actual)', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix - Phần trăm (Normalized)', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Lưu figure
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu confusion matrix tại: {output_path}")
    
    # Cũng lưu dạng số lớn hơn
    output_path_large = os.path.join(output_dir, 'confusion_matrix_large.png')
    fig_large, ax_large = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_large,
        cbar_kws={'label': 'Phần trăm (%)'},
        annot_kws={'size': 12}
    )
    ax_large.set_xlabel('Dự đoán (Predicted)', fontsize=14, fontweight='bold')
    ax_large.set_ylabel('Thực tế (Actual)', fontsize=14, fontweight='bold')
    ax_large.set_title('Confusion Matrix - Phần trăm', fontsize=16, fontweight='bold')
    ax_large.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path_large, dpi=300, bbox_inches='tight')
    print(f"Đã lưu confusion matrix (large) tại: {output_path_large}")
    plt.close()
    
    return cm, cm_normalized

def calculate_metrics(ground_truth, predictions, class_names):
    """
    Tính toán các metrics đánh giá
    """
    print("\n=== Tính toán các metrics đánh giá ===")
    
    # Classification report
    report = classification_report(
        ground_truth,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # In ra metrics cho từng class
    print("\n=== Metrics theo từng class ===")
    for i, class_name in enumerate(class_names):
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(f"\n{class_name}:")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  F1-score: {f1:.4f} ({f1*100:.2f}%)")
            print(f"  Support: {int(support)}")
    
    # Overall metrics
    print("\n=== Overall Metrics ===")
    print(f"Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"Macro avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted avg Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Weighted avg Recall: {report['weighted avg']['recall']:.4f}")
    print(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")
    
    return report

def save_evaluation_report(ground_truth, predictions, class_names, cm, cm_normalized, report, output_dir):
    """
    Lưu báo cáo đánh giá ra file text
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH YOLOV8 MULTICLASS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("THÔNG TIN MÔ HÌNH:\n")
        f.write(f"- Model path: {MODEL_PATH}\n")
        f.write(f"- Data config: {DATA_YAML}\n")
        f.write(f"- Test images: {TEST_IMAGES_DIR}\n\n")
        
        f.write("THỐNG KÊ DỮ LIỆU:\n")
        f.write(f"- Tổng số objects trong ground truth: {len(ground_truth)}\n")
        f.write(f"- Tổng số objects được phát hiện: {len(predictions)}\n")
        f.write(f"- Số lớp: {len(class_names)}\n\n")
        
        f.write("DANH SÁCH CÁC LỚP:\n")
        for i, class_name in enumerate(class_names):
            gt_count = ground_truth.count(i)
            pred_count = predictions.count(i)
            f.write(f"  {i}. {class_name}: GT={gt_count}, Pred={pred_count}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX (Số lượng tuyệt đối)\n")
        f.write("=" * 80 + "\n")
        f.write("Actual \\ Predicted\t" + "\t".join(class_names) + "\n")
        for i, class_name in enumerate(class_names):
            row = f"{class_name}\t\t"
            row += "\t".join([str(int(cm[i, j])) for j in range(len(class_names))])
            f.write(row + "\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX (Phần trăm)\n")
        f.write("=" * 80 + "\n")
        f.write("Actual \\ Predicted\t" + "\t".join(class_names) + "\n")
        for i, class_name in enumerate(class_names):
            row = f"{class_name}\t\t"
            row += "\t".join([f"{cm_normalized[i, j]*100:.2f}%" for j in range(len(class_names))])
            f.write(row + "\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("METRICS ĐÁNH GIÁ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Metrics theo từng class:\n")
        for i, class_name in enumerate(class_names):
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {precision:.4f} ({precision*100:.2f}%)\n")
                f.write(f"  Recall: {recall:.4f} ({recall*100:.2f}%)\n")
                f.write(f"  F1-score: {f1:.4f} ({f1*100:.2f}%)\n")
                f.write(f"  Support: {int(support)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Overall Metrics:\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)\n")
        f.write(f"Macro avg Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"Macro avg Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"Macro avg F1-score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted avg Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted avg Recall: {report['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"\nĐã lưu báo cáo đánh giá tại: {report_path}")

def evaluate_with_yolo_val(model_path, data_yaml_path, output_dir):
    """
    Sử dụng YOLOv8 built-in validation để đánh giá và tạo confusion matrix
    """
    print("\n=== Sử dụng YOLOv8 built-in validation ===")
    
    # Load model
    print(f"Đang load mô hình từ: {model_path}")
    model = YOLO(model_path)
    
    # Load data config
    data_config = load_data_config(data_yaml_path)
    class_names = data_config['names']
    
    # Đánh giá trên test set
    print("\nĐang đánh giá mô hình trên test set...")
    results = model.val(
        data=data_yaml_path,
        split='test',  # Sử dụng test split
        conf=0.25,
        iou=0.45,
        save_json=False,
        plots=True,
        save_dir=output_dir
    )
    
    print("\n=== Kết quả đánh giá ===")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    
    # Tìm confusion matrix file được tạo bởi YOLOv8
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    
    if os.path.exists(confusion_matrix_path):
        print(f"\nConfusion matrix đã được tạo tại: {confusion_matrix_path}")
        # Copy và tạo bản sao với tên khác
        import shutil
        shutil.copy(confusion_matrix_path, 
                   os.path.join(output_dir, 'confusion_matrix_yolo.png'))
    else:
        print("\nKhông tìm thấy confusion matrix được tạo tự động.")
        print("Sẽ tạo confusion matrix thủ công...")
    
    return results, class_names

def main():
    """Hàm chính"""
    print("=" * 80)
    print("ĐÁNH GIÁ MÔ HÌNH YOLOV8 MULTICLASS")
    print("=" * 80)
    print()
    
    # Kiểm tra file tồn tại
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Không tìm thấy model tại: {MODEL_PATH}")
        return
    
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: Không tìm thấy data.yaml tại: {DATA_YAML}")
        return
    
    # Sử dụng YOLOv8 built-in validation (cách 1)
    try:
        results, class_names = evaluate_with_yolo_val(
            MODEL_PATH,
            DATA_YAML,
            OUTPUT_DIR
        )
        
        # Lưu metrics vào file
        metrics_path = os.path.join(OUTPUT_DIR, 'metrics_summary.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH YOLOV8 MULTICLASS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"Data config: {DATA_YAML}\n\n")
            f.write("METRICS:\n")
            f.write(f"Precision (B): {results.results_dict.get('metrics/precision(B)', 0):.4f}\n")
            f.write(f"Recall (B): {results.results_dict.get('metrics/recall(B)', 0):.4f}\n")
            f.write(f"mAP50 (B): {results.results_dict.get('metrics/mAP50(B)', 0):.4f}\n")
            f.write(f"mAP50-95 (B): {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}\n")
        print(f"\nĐã lưu metrics tại: {metrics_path}")
        
    except Exception as e:
        print(f"\nLỗi khi sử dụng YOLOv8 validation: {e}")
        print("Sẽ thử phương pháp thủ công...")
    
    # Đánh giá thủ công để tạo confusion matrix chi tiết (cách 2)
    print("\n" + "=" * 80)
    print("ĐÁNH GIÁ THỦ CÔNG ĐỂ TẠO CONFUSION MATRIX CHI TIẾT")
    print("=" * 80)
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"WARNING: Không tìm thấy thư mục test images tại: {TEST_IMAGES_DIR}")
        print("Bỏ qua đánh giá thủ công.")
    else:
        try:
            ground_truth, predictions, class_names, num_classes = evaluate_model_on_test(
                MODEL_PATH,
                DATA_YAML,
                TEST_IMAGES_DIR,
                conf_threshold=0.25,
                iou_threshold=0.45
            )
            
            if len(ground_truth) > 0:
                # Tạo confusion matrix
                cm, cm_normalized = create_confusion_matrix(
                    ground_truth,
                    predictions,
                    class_names,
                    OUTPUT_DIR
                )
                
                # Tính toán metrics
                report = calculate_metrics(ground_truth, predictions, class_names)
                
                # Lưu báo cáo
                save_evaluation_report(
                    ground_truth,
                    predictions,
                    class_names,
                    cm,
                    cm_normalized,
                    report,
                    OUTPUT_DIR
                )
            else:
                print("WARNING: Không tìm thấy ground truth labels!")
        except Exception as e:
            print(f"Lỗi khi đánh giá thủ công: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("HOÀN TẤT ĐÁNH GIÁ!")
    print("=" * 80)
    print(f"Kết quả đã được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
