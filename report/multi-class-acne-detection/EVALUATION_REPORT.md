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
- **Training Box Loss**: 2.22527
- **Training Class Loss**: 2.17519
- **Training DFL Loss**: 1.33678
- **Validation Box Loss**: 2.27779
- **Validation Class Loss**: 2.43256
- **Validation DFL Loss**: 1.32198

**Phân Tích Epoch Tốt Nhất:**
- ⚠️ **mAP@0.5 chỉ đạt 6.68%** - Kết quả rất thấp, cho thấy mô hình chưa học tốt
- ⚠️ **Precision và Recall đều thấp**: 18.37% và 11.21% - Mô hình gặp khó khăn trong việc phát hiện và phân loại mụn
- ⚠️ **mAP@0.5:0.95 rất thấp**: 2.27% - Độ chính xác vị trí bounding box rất kém
- ⚠️ **Validation loss cao hơn training loss**: Có dấu hiệu overfitting

**So Sánh với Epoch Cuối (Epoch 20):**
| Metric | Epoch 18 (Best) | Epoch 20 (Final) | Chênh Lệch |
|--------|----------------|------------------|------------|
| Precision | 18.37% (0.1837) | 14.92% (0.14918) | -3.45% ⬇️ |
| Recall | 11.21% (0.11205) | 9.86% (0.09858) | -1.35% ⬇️ |
| mAP@0.5 | **6.68% (0.06679)** | 5.84% (0.05841) | -0.84% ⬇️ |
| mAP@0.5:0.95 | 2.27% (0.02269) | 2.00% (0.02002) | -0.27% ⬇️ |

**Nhận Xét:** Epoch 18 cho kết quả tốt nhất nhưng vẫn rất thấp. Epoch 20 có kết quả thấp hơn, cho thấy mô hình có thể đã bị overfitting hoặc chưa học đủ.

### 2.2. Metrics Tổng Quan

#### Epoch Cuối Cùng (Epoch 20)
| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.14918 | 14.92% |
| **Recall** | 0.09858 | 9.86% |
| **mAP@0.5** | 0.05841 | 5.84% |
| **mAP@0.5:0.95** | 0.02002 | 2.00% |

#### Loss Values (Epoch 20)
- **Training Box Loss**: 2.22071
- **Training Class Loss**: 2.13984
- **Training DFL Loss**: 1.30604
- **Validation Box Loss**: 2.30100
- **Validation Class Loss**: 2.41682
- **Validation DFL Loss**: 1.32401

### 2.3. Phân Tích Xu Hướng Huấn Luyện

#### Epoch Đầu (Epoch 1)
- Precision: 2.23% (0.02233)
- Recall: 21.26% (0.21261)
- mAP@0.5: 2.55% (0.02554)
- mAP@0.5:0.95: 0.93% (0.00933)

#### Các Epoch Đáng Chú Ý
- **Epoch 18**: mAP@0.5 cao nhất (6.68% - 0.06679) ⭐
- **Epoch 8**: Precision cao nhất (25.18% - 0.25184)
- **Epoch 1**: Recall cao nhất (21.26% - 0.21261)
- **Epoch 17**: mAP@0.5 tốt (6.57% - 0.06572)

#### Cải Thiện Qua Quá Trình Huấn Luyện
- **Precision**: Tăng từ 2.23% (0.02233) → 14.92% (0.14918) (+12.69%) - Cải thiện đáng kể nhưng vẫn thấp
- **Recall**: Giảm từ 21.26% (0.21261) → 9.86% (0.09858) (-11.40%) - Giảm đáng kể, mô hình bỏ sót nhiều mụn
- **mAP@0.5**: Tăng từ 2.55% (0.02554) → 5.84% (0.05841) (+3.29%) - Cải thiện nhỏ
- **mAP@0.5:0.95**: Tăng từ 0.93% (0.00933) → 2.00% (0.02002) (+1.07%) - Cải thiện rất nhỏ

**Nhận Xét Quan Trọng:**
- ⚠️ **Recall giảm mạnh**: Từ 21.26% xuống 9.86% - Mô hình ngày càng bỏ sót nhiều mụn
- ⚠️ **Precision tăng nhưng vẫn thấp**: Từ 2.23% lên 14.92% - Mô hình học được một số đặc điểm nhưng chưa đủ
- ⚠️ **mAP@0.5 rất thấp**: Chỉ đạt 6.68% ở epoch tốt nhất - Mô hình chưa đủ khả năng phát hiện mụn

### 2.4. Thời Gian Huấn Luyện

**Tổng thời gian training:**
- **Epoch 20 (cuối cùng)**: 3454.41 giây (≈57.6 phút)
- **Thời gian trung bình mỗi epoch**: 172.7 giây (≈2.9 phút/epoch)
- **Tốc độ training**: ~0.35 epochs/phút

**Thời gian các epoch:**
- Epoch 1: 164.26 giây (≈2.7 phút)
- Epoch 10: 1598.25 giây (≈26.6 phút) - Cumulative
- Epoch 18 (best): 3081.68 giây (≈51.4 phút) - Cumulative
- Epoch 20 (final): 3454.41 giây (≈57.6 phút) - Cumulative

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

## 3.4. Kết Quả Đánh Giá Trên Test Set

### 3.4.1. Tổng Quan Test Set

**Thông tin Test Set:**
- **Số lượng ảnh**: 92 images
- **Tổng số instances (objects)**: 923
- **Số lớp**: 6 classes
- **Cache**: Đã tạo cache mới tại `/Users/quangthai/Documents/AI in Bioinfomatics/acne2/data-2 copy/test/labels.cache`

### 3.4.2. Kết Quả Validation Trên Test Set

**Metrics Tổng Quán (All Classes):**
| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision (Box)** | 0.296 | **29.6%** |
| **Recall** | 0.0814 | **8.14%** |
| **mAP@0.5** | 0.18 | **18.0%** |
| **mAP@0.5:0.95** | 0.0735 | **7.35%** |

**Nhận xét:**
- ⚠️ **Precision: 29.6%** - Thấp, có nhiều false positives
- ⚠️ **Recall: 8.14%** - Rất thấp, mô hình bỏ sót hơn 90% objects
- ⚠️ **mAP@0.5: 18.0%** - Thấp, cho thấy mô hình gặp khó khăn trong phát hiện
- ⚠️ **mAP@0.5:0.95: 7.35%** - Rất thấp, độ chính xác vị trí bounding box kém

### 3.4.3. Kết Quả Chi Tiết Theo Từng Class

Bảng dưới đây trình bày hiệu suất của mô hình cho từng loại mụn trên test set:

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| **all** | 92 | 923 | 0.296 | 0.0814 | 0.18 | 0.0735 |
| **blackheads** | 10 | 34 | 0.0 | 0.0 | 0.0 | 0.0 |
| **dark spot** | 49 | 385 | 0.676 | 0.0597 | 0.367 | 0.154 |
| **nodules** | 17 | 46 | 0.0 | 0.0 | 0.0 | 0.0 |
| **papules** | 45 | 185 | 0.509 | 0.146 | 0.306 | 0.14 |
| **pustules** | 32 | 129 | 0.319 | 0.178 | 0.252 | 0.103 |
| **whiteheads** | 22 | 144 | 0.273 | 0.104 | 0.158 | 0.0432 |
| **Macro Avg** | - | **923** | **0.296** | **0.0813** | **0.181** | **0.0734** |
| **Weighted Avg** | - | **923** | **0.296** | **0.0814** | **0.18** | **0.0735** |

**Ghi chú:**
- **Macro Avg**: Trung bình đơn giản của tất cả các lớp (không trọng số)
- **Weighted Avg**: Trung bình có trọng số theo số lượng instances của mỗi lớp
- Công thức Macro Avg: (blackheads + dark spot + nodules + papules + pustules + whiteheads) / 6
- Công thức Weighted Avg: Σ(metric × instances) / tổng instances

### 3.4.4. Phân Tích Chi Tiết Từng Class

#### 1. Dark Spot (Vết thâm) - ⭐⭐⭐ Tốt Nhất
- **Precision: 67.6%** - Cao nhất trong tất cả các lớp
- **Recall: 5.97%** - Rất thấp, bỏ sót hơn 94% vết thâm
- **mAP@0.5: 36.7%** - Tốt nhất trong tất cả các lớp
- **mAP@0.5:0.95: 15.4%** - Tốt nhất
- **Support**: 49 images, 385 instances (lớp có nhiều instances nhất)
- **Nhận xét**: Mặc dù precision cao nhưng recall rất thấp, mô hình phát hiện đúng nhưng bỏ sót rất nhiều

#### 2. Papules (Mụn sần) - ⭐⭐
- **Precision: 50.9%** - Khá tốt
- **Recall: 14.6%** - Thấp, bỏ sót hơn 85% mụn sần
- **mAP@0.5: 30.6%** - Đứng thứ 2
- **mAP@0.5:0.95: 14.0%** - Đứng thứ 2
- **Support**: 45 images, 185 instances
- **Nhận xét**: Precision khá tốt nhưng recall rất thấp

#### 3. Pustules (Mụn mủ) - ⭐⭐
- **Precision: 31.9%** - Trung bình
- **Recall: 17.8%** - Thấp, nhưng cao nhất trong các lớp (vẫn bỏ sót hơn 82%)
- **mAP@0.5: 25.2%** - Đứng thứ 3
- **mAP@0.5:0.95: 10.3%** - Đứng thứ 3
- **Support**: 32 images, 129 instances
- **Nhận xét**: Recall cao nhất nhưng vẫn rất thấp, precision trung bình

#### 4. Whiteheads (Mụn đầu trắng) - ⭐
- **Precision: 27.3%** - Thấp
- **Recall: 10.4%** - Rất thấp
- **mAP@0.5: 15.8%** - Thấp
- **mAP@0.5:0.95: 4.32%** - Rất thấp
- **Support**: 22 images, 144 instances
- **Nhận xét**: Hiệu suất thấp, cần cải thiện

#### 5. Blackheads (Mụn đầu đen) - ❌ Không Phát Hiện Được
- **Precision: 0.0%** - Không phát hiện được
- **Recall: 0.0%** - Không phát hiện được
- **mAP@0.5: 0.0%** - Không phát hiện được
- **mAP@0.5:0.95: 0.0%** - Không phát hiện được
- **Support**: 10 images, 34 instances
- **⚠️ Vấn đề nghiêm trọng**: Mô hình hoàn toàn không phát hiện được blackheads trên test set

#### 6. Nodules (Mụn nang) - ❌ Không Phát Hiện Được
- **Precision: 0.0%** - Không phát hiện được
- **Recall: 0.0%** - Không phát hiện được
- **mAP@0.5: 0.0%** - Không phát hiện được
- **mAP@0.5:0.95: 0.0%** - Không phát hiện được
- **Support**: 17 images, 46 instances
- **⚠️ Vấn đề nghiêm trọng**: Mô hình hoàn toàn không phát hiện được nodules trên test set

### 3.4.5. So Sánh với Kết Quả Training

| Metric | Training (Epoch 18) | Test Set | Chênh Lệch |
|--------|---------------------|----------|------------|
| **Precision** | 18.37% | 29.6% | +11.23% ⬆️ |
| **Recall** | 11.21% | 8.14% | -3.07% ⬇️ |
| **mAP@0.5** | 6.68% | 18.0% | +11.32% ⬆️ |
| **mAP@0.5:0.95** | 2.27% | 7.35% | +5.08% ⬆️ |

**Nhận xét:**
- ✅ **Precision và mAP tốt hơn trên test set** - Mô hình có precision cao hơn khi đánh giá trên test set
- ⚠️ **Recall thấp hơn trên test set** - Mô hình bỏ sót nhiều hơn trên test set
- ⚠️ **2/6 lớp không phát hiện được** (blackheads, nodules) - Vấn đề nghiêm trọng
- ⚠️ **Tất cả các lớp đều có recall rất thấp** (< 18%) - Mô hình bỏ sót phần lớn objects

### 3.4.6. Kết Luận

**Điểm mạnh:**
- ✅ Dark spot có precision và mAP cao nhất (67.6%, 36.7%)
- ✅ Precision tổng thể tốt hơn trên test set (29.6% vs 18.37%)

**Điểm yếu nghiêm trọng:**
- ❌ **Recall rất thấp (8.14%)** - Mô hình bỏ sót hơn 90% objects
- ❌ **2 lớp không phát hiện được**: Blackheads và Nodules
- ❌ **Tất cả các lớp đều có recall < 18%** - Mô hình gặp khó khăn lớn trong việc phát hiện objects
- ❌ **mAP@0.5:0.95 rất thấp (7.35%)** - Độ chính xác vị trí bounding box kém

**Khuyến nghị:**
1. **Ưu tiên cao**: Cải thiện recall - mô hình đang bỏ sót quá nhiều objects
2. **Khắc phục ngay**: Tập trung vào blackheads và nodules - 2 lớp không phát hiện được
3. **Cải thiện overall**: Tăng số epochs, sử dụng model lớn hơn, tăng cường data augmentation

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

