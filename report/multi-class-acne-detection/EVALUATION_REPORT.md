# Báo Cáo Đánh Giá Kết Quả Mô Hình Phát Hiện Mụn Đa Lớp (Multi-Class Acne Detection)

## 1. Tổng Quan

### 1.1. Mô Tả Dataset

#### Nguồn Dữ Liệu
- **Nguồn**: Roboflow Universe
- **Workspace**: kritsakorn
- **Project**: acne-kbm0q
- **Version**: 20
- **License**: Private
- **URL**: https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/20

#### Cấu Trúc Dataset
Dataset được tổ chức theo định dạng YOLO với cấu trúc:
```
Acne-20/
├── train/
│   ├── images/     (982 ảnh)
│   └── labels/     (982 file annotation)
├── valid/
│   ├── images/     (197 ảnh)
│   └── labels/     (197 file annotation)
└── test/
    ├── images/     (92 ảnh)
    └── labels/     (92 file annotation)
```

#### Thống Kê Dataset
| Split | Số Lượng Ảnh | Tỷ Lệ | Mô Tả |
|-------|--------------|-------|-------|
| **Training** | 982 | ~77.2% | Dữ liệu huấn luyện |
| **Validation** | 197 | ~15.5% | Dữ liệu đánh giá trong quá trình huấn luyện |
| **Test** | 92 | ~7.2% | Dữ liệu kiểm tra cuối cùng |
| **Tổng cộng** | **1,271** | 100% | Tổng số ảnh trong dataset |

#### Phân Loại (Classes)
Dataset bao gồm **6 loại mụn** khác nhau:

| ID | Tên Lớp | Tên Tiếng Anh | Mô Tả |
|----|---------|---------------|-------|
| 0 | Mụn đầu đen | **Blackheads** | Mụn có đầu đen, thường xuất hiện ở mũi và vùng chữ T |
| 1 | Vết thâm | **Dark spot** | Vết thâm, đốm nâu sau khi mụn lành |
| 2 | Mụn nang | **Nodules** | Mụn viêm sâu, cứng, đau, kích thước lớn |
| 3 | Mụn sần | **Papules** | Mụn viêm nhỏ, đỏ, không có mủ |
| 4 | Mụn mủ | **Pustules** | Mụn viêm có đầu mủ trắng hoặc vàng |
| 5 | Mụn đầu trắng | **Whiteheads** | Mụn có đầu trắng, tương tự blackheads nhưng màu trắng |

**Tổng số lớp (nc)**: 6

### 1.2. Thông Tin Mô Hình
- **Kiến trúc**: YOLOv8n (Nano) - Model nhỏ nhất, tốc độ nhanh nhất
- **Nhiệm vụ**: Object Detection - Phát hiện và phân loại mụn đa lớp
- **Số lớp**: 6 loại mụn (như mô tả ở trên)

### 1.3. Cấu Hình Huấn Luyện
- **Epochs**: 20 (đã hoàn thành)
- **Batch size**: 16
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.001 (initial)
- **Device**: MPS (Mac GPU)
- **Data augmentation**: 
  - Mosaic: 1.0
  - HSV augmentation
  - Translation: 0.1
  - Scale: 0.5
  - Flip LR: 0.5
  - RandAugment: enabled
  - Erasing: 0.4
  - **Lưu ý**: Không sử dụng rotation, shear, mixup (degrees: 0.0, shear: 0.0, mixup: 0.0)

## 2. Kết Quả Huấn Luyện

### 2.1. Kết Quả Tốt Nhất Đạt Được

#### Epoch Tốt Nhất: Epoch 18 (Best mAP@0.5)

**Metrics Chính:**
| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.1837 | **18.37%** |
| **Recall** | 0.1121 | **11.21%** |
| **mAP@0.5** | 0.0668 | **6.68%** |
| **mAP@0.5:0.95** | 0.0227 | **2.27%** |

**Loss Values (Epoch 18):**
- **Training Box Loss**: 2.2253
- **Training Class Loss**: 2.1752
- **Training DFL Loss**: 1.3368
- **Validation Box Loss**: 2.2778
- **Validation Class Loss**: 2.4326
- **Validation DFL Loss**: 1.3220

**Phân Tích Epoch Tốt Nhất:**
- ⚠️ **mAP@0.5 chỉ đạt 6.68%** - Kết quả rất thấp, cho thấy mô hình chưa học tốt
- ⚠️ **Precision và Recall đều thấp**: 18.37% và 11.21% - Mô hình gặp khó khăn trong việc phát hiện và phân loại mụn
- ⚠️ **mAP@0.5:0.95 rất thấp**: 2.27% - Độ chính xác vị trí bounding box rất kém
- ⚠️ **Validation loss cao hơn training loss**: Có dấu hiệu overfitting

**So Sánh với Epoch Cuối (Epoch 20):**
| Metric | Epoch 18 (Best) | Epoch 20 (Final) | Chênh Lệch |
|--------|----------------|------------------|------------|
| Precision | 18.37% | 14.92% | -3.45% ⬇️ |
| Recall | 11.21% | 9.86% | -1.35% ⬇️ |
| mAP@0.5 | **6.68%** | 5.84% | -0.84% ⬇️ |
| mAP@0.5:0.95 | 2.27% | 2.00% | -0.27% ⬇️ |

**Nhận Xét:** Epoch 18 cho kết quả tốt nhất nhưng vẫn rất thấp. Epoch 20 có kết quả thấp hơn, cho thấy mô hình có thể đã bị overfitting hoặc chưa học đủ.

### 2.2. Metrics Tổng Quan

#### Epoch Cuối Cùng (Epoch 20)
| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.1492 | 14.92% |
| **Recall** | 0.0986 | 9.86% |
| **mAP@0.5** | 0.0584 | 5.84% |
| **mAP@0.5:0.95** | 0.0200 | 2.00% |

#### Loss Values (Epoch 20)
- **Training Box Loss**: 2.2207
- **Training Class Loss**: 2.1398
- **Training DFL Loss**: 1.3060
- **Validation Box Loss**: 2.3010
- **Validation Class Loss**: 2.4168
- **Validation DFL Loss**: 1.3240

### 2.3. Phân Tích Xu Hướng Huấn Luyện

#### Epoch Đầu (Epoch 1)
- Precision: 2.23%
- Recall: 21.26%
- mAP@0.5: 2.55%
- mAP@0.5:0.95: 0.93%

#### Các Epoch Đáng Chú Ý
- **Epoch 18**: mAP@0.5 cao nhất (6.68%) ⭐
- **Epoch 8**: Precision cao nhất (25.18%)
- **Epoch 1**: Recall cao nhất (21.26%)
- **Epoch 17**: mAP@0.5 tốt (6.57%)

#### Cải Thiện Qua Quá Trình Huấn Luyện
- **Precision**: Tăng từ 2.23% → 14.92% (+12.69%) - Cải thiện đáng kể nhưng vẫn thấp
- **Recall**: Giảm từ 21.26% → 9.86% (-11.40%) - Giảm đáng kể, mô hình bỏ sót nhiều mụn
- **mAP@0.5**: Tăng từ 2.55% → 5.84% (+3.29%) - Cải thiện nhỏ
- **mAP@0.5:0.95**: Tăng từ 0.93% → 2.00% (+1.07%) - Cải thiện rất nhỏ

**Nhận Xét Quan Trọng:**
- ⚠️ **Recall giảm mạnh**: Từ 21.26% xuống 9.86% - Mô hình ngày càng bỏ sót nhiều mụn
- ⚠️ **Precision tăng nhưng vẫn thấp**: Từ 2.23% lên 14.92% - Mô hình học được một số đặc điểm nhưng chưa đủ
- ⚠️ **mAP@0.5 rất thấp**: Chỉ đạt 6.68% ở epoch tốt nhất - Mô hình chưa đủ khả năng phát hiện mụn

### 2.4. Thời Gian Huấn Luyện
- **Tổng thời gian**: ~3454.41 giây (≈57.6 phút)
- **Thời gian trung bình mỗi epoch**: ~172.7 giây (≈2.9 phút)
- **Tốc độ**: ~0.35 epochs/phút

## 3. Phân Tích Ma Trận Nhầm Lẫn (Confusion Matrix)

### 3.1. Tổng Quan về Ma Trận Nhầm Lẫn

Ma trận nhầm lẫn (Confusion Matrix) cho thấy hiệu suất phân loại của mô hình cho 6 loại mụn khác nhau.

**File tham khảo:**
- `acne_yolov8_multiclass/labels.jpg` - Visualization của labels trong dataset

### 3.2. Phân Tích Dựa trên Metrics Tổng Quan

Dựa trên kết quả mAP@0.5 = 6.68% và Precision = 18.37%, có thể suy ra:

#### Vấn Đề Nghiêm Trọng
- **Precision thấp (18.37%)**: Mô hình có nhiều false positives - dự đoán sai nhiều
- **Recall rất thấp (11.21%)**: Mô hình bỏ sót rất nhiều mụn thực tế
- **mAP@0.5 rất thấp (6.68%)**: Mô hình gặp khó khăn lớn trong việc phát hiện và phân loại mụn

#### Các Loại Lỗi Thường Gặp
1. **False Positives (Dự đoán sai)**: Rất nhiều - Mô hình dự đoán có mụn nhưng thực tế không có hoặc nhầm loại
2. **False Negatives (Bỏ sót)**: Rất nhiều - Mô hình không phát hiện được phần lớn mụn thực tế
3. **Misclassification (Nhầm lẫn giữa các lớp)**: Có thể xảy ra nhiều do mAP thấp

### 3.3. Khuyến Nghị Dựa trên Confusion Matrix

**Vấn đề cần giải quyết ngay:**

1. **Tăng số lượng epochs**: 20 epochs là quá ít cho bài toán phức tạp này
2. **Sử dụng model lớn hơn**: YOLOv8n quá nhỏ, nên thử YOLOv8s hoặc YOLOv8m
3. **Điều chỉnh learning rate**: Có thể learning rate 0.001 chưa phù hợp
4. **Tăng cường data augmentation**: Thêm rotation, shear, mixup để mô hình học tốt hơn
5. **Kiểm tra chất lượng dataset**: Đảm bảo annotation chính xác

## 4. Đánh Giá Chi Tiết

### 4.1. Điểm Mạnh
1. **Tốc độ training nhanh**: YOLOv8n nhỏ, training nhanh (~2.9 phút/epoch)
2. **Có cải thiện qua các epochs**: Mặc dù thấp nhưng vẫn có xu hướng cải thiện
3. **Model nhẹ**: Phù hợp cho deployment trên thiết bị có tài nguyên hạn chế

### 4.2. Điểm Yếu và Hạn Chế
1. **Hiệu suất rất thấp**: mAP@0.5 chỉ 6.68% - Không đủ để sử dụng thực tế
2. **Recall giảm mạnh**: Từ 21.26% xuống 9.86% - Mô hình ngày càng bỏ sót nhiều
3. **Precision thấp**: 18.37% - Nhiều dự đoán sai
4. **Số epochs quá ít**: 20 epochs không đủ cho mô hình học tốt
5. **Model quá nhỏ**: YOLOv8n có thể không đủ capacity cho bài toán 6 lớp
6. **Validation loss cao**: Có dấu hiệu overfitting hoặc chưa học đủ
7. **Data augmentation hạn chế**: Không sử dụng rotation, shear, mixup

### 4.3. Phân Tích Loss
- **Training losses giảm chậm**: Box loss, class loss và DFL loss giảm nhưng chậm
- **Validation losses cao và dao động**: Validation loss cao hơn training loss, cho thấy vấn đề
- **Class loss cao**: 2.14-2.43 cho thấy mô hình gặp khó khăn trong phân loại
- **Box loss cao**: 2.22-2.30 cho thấy mô hình gặp khó khăn trong localization

## 5. Khuyến Nghị Cải Thiện

### 5.1. Ngắn Hạn (Ưu Tiên Cao)
1. **Tăng số epochs**: Từ 20 lên ít nhất 100-200 epochs
2. **Sử dụng model lớn hơn**: Chuyển từ YOLOv8n sang YOLOv8s hoặc YOLOv8m
3. **Tăng cường data augmentation**:
   - Thêm rotation (degrees: 10-15°)
   - Thêm shear (shear: 5-10°)
   - Thêm mixup (mixup: 0.1-0.2)
4. **Điều chỉnh learning rate**: Thử learning rate thấp hơn (0.0005) hoặc sử dụng cosine annealing
5. **Early stopping**: Sử dụng patience để tránh overfitting

### 5.2. Trung Hạn
1. **Kiểm tra và cải thiện dataset**:
   - Xem xét lại chất lượng annotation
   - Kiểm tra cân bằng dữ liệu giữa các lớp
   - Thêm dữ liệu nếu cần
2. **Tối ưu hóa hyperparameters**:
   - Điều chỉnh loss weights (box, cls, dfl)
   - Thử nghiệm với các optimizer khác
   - Điều chỉnh batch size
3. **Transfer learning tốt hơn**: Sử dụng pre-trained weights từ dataset tương tự

### 5.3. Dài Hạn
1. **Mở rộng dataset**: Thu thập thêm dữ liệu, đặc biệt là các trường hợp khó
2. **Ensemble methods**: Kết hợp nhiều mô hình
3. **Sử dụng YOLOv11 hoặc các model mới hơn**: Thử nghiệm với các kiến trúc mới
4. **Fine-tuning chi tiết**: Fine-tune từng phần của model

## 6. So Sánh với Benchmark

### 6.1. Tiêu Chuẩn Đánh Giá
- **mAP@0.5 > 50%**: ❌ Chưa đạt (6.68%)
- **mAP@0.5:0.95 > 30%**: ❌ Chưa đạt (2.27%)
- **Precision > 60%**: ❌ Chưa đạt (18.37%)
- **Recall > 60%**: ❌ Chưa đạt (11.21%)

### 6.2. So Sánh với Model Trước (acne_base)
| Metric | acne_base (YOLOv8m) | acne_yolov8_multiclass (YOLOv8n) | Chênh Lệch |
|--------|---------------------|----------------------------------|------------|
| Model Size | Medium | Nano | Nhỏ hơn |
| Epochs | 85/200 | 20/20 | Ít hơn |
| mAP@0.5 (Best) | 63.00% | 6.68% | -56.32% ⬇️ |
| Precision (Best) | 63.14% | 18.37% | -44.77% ⬇️ |
| Recall (Best) | 59.40% | 11.21% | -48.19% ⬇️ |
| mAP@0.5:0.95 (Best) | 28.34% | 2.27% | -26.07% ⬇️ |

**Nhận Xét:** Model này có hiệu suất thấp hơn rất nhiều so với model trước, chủ yếu do:
- Model nhỏ hơn (nano vs medium)
- Ít epochs hơn (20 vs 85)
- Có thể do cấu hình training khác nhau

### 6.3. Đánh Giá Tổng Thể
Mô hình **chưa đạt yêu cầu** cho bài toán phát hiện mụn:
- ❌ Hiệu suất quá thấp để sử dụng thực tế
- ❌ Cần cải thiện đáng kể trước khi deploy
- ⚠️ Có tiềm năng cải thiện với các thay đổi được đề xuất

## 7. Kết Luận

### 7.1. Tóm Tắt Kết Quả

Mô hình YOLOv8n đã được huấn luyện cho bài toán phát hiện và phân loại 6 loại mụn, nhưng **kết quả chưa đạt yêu cầu**.

**Kết quả tốt nhất đạt được (Epoch 18):**
- ❌ **mAP@0.5: 6.68%** - Rất thấp, không đủ để sử dụng
- ❌ **Precision: 18.37%** - Nhiều dự đoán sai
- ❌ **Recall: 11.21%** - Bỏ sót rất nhiều mụn
- ❌ **mAP@0.5:0.95: 2.27%** - Độ chính xác vị trí rất kém

**Kết quả epoch cuối (Epoch 20):**
- mAP@0.5: 5.84% (thấp hơn epoch tốt nhất)
- Có dấu hiệu chưa học đủ hoặc overfitting

### 7.2. Đánh Giá Tổng Thể

- ✅ Training nhanh
- ✅ Model nhẹ, phù hợp deployment
- ✅ Có xu hướng cải thiện (dù nhỏ)

**Điểm cần cải thiện:**
- ❌ Hiệu suất quá thấp (mAP@0.5: 6.68%)
- ❌ Không đủ để sử dụng thực tế
- ❌ Cần thay đổi đáng kể về model và cấu hình

**Kết luận:** Mô hình **chưa sẵn sàng** để sử dụng trong môi trường thực tế. Cần thực hiện các cải thiện được đề xuất, đặc biệt là:
1. Sử dụng model lớn hơn (YOLOv8s hoặc YOLOv8m)
2. Tăng số epochs lên ít nhất 100-200
3. Tăng cường data augmentation
4. Điều chỉnh hyperparameters

---

**Ngày tạo báo cáo**: 2024-12-13  
**Thư mục kết quả**: `acne_yolov8_multiclass/`  
**File kết quả**: 
- `results.csv` - Metrics chi tiết qua các epochs
- `labels.jpg` - Visualization của labels
- `weights/best.pt` - Model weights tốt nhất (Epoch 18)
- `weights/last.pt` - Model weights epoch cuối cùng
- `weights/best.onnx` - Model ONNX format
- `weights/best.torchscript` - Model TorchScript format

