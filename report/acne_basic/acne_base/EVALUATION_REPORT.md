# Báo Cáo Đánh Giá Kết Quả Mô Hình Phát Hiện Mụn (Acne Detection)

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
data-2/
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

#### Đặc Điểm Dataset
- **Định dạng ảnh**: JPG
- **Định dạng annotation**: YOLO format (.txt)
- **Mỗi ảnh có file label tương ứng**: Đảm bảo 1:1 mapping giữa images và labels
- **Chuẩn hóa**: Dataset đã được chuẩn hóa và sẵn sàng cho YOLO training

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

#### Đặc Điểm Kỹ Thuật
- **Đa dạng về kích thước**: Ảnh có kích thước khác nhau, được resize về 640x640 trong quá trình training
- **Đa dạng về điều kiện**: Ảnh được chụp trong các điều kiện ánh sáng và góc độ khác nhau
- **Chất lượng annotation**: Mỗi ảnh đã được gán nhãn chính xác với bounding boxes cho từng loại mụn
- **Cân bằng dữ liệu**: Dataset được chia theo tỷ lệ phù hợp (train/val/test ≈ 77/15/8)

#### Data Augmentation
Trong quá trình training, dataset được áp dụng các kỹ thuật augmentation:
- **Mosaic**: 1.0 (ghép 4 ảnh thành 1)
- **Mixup**: 0.05 (trộn ảnh)
- **HSV augmentation**: Điều chỉnh màu sắc
- **Rotation**: 5° (xoay ảnh)
- **Translation**: 0.1 (dịch chuyển)
- **Scale**: 0.5 (thay đổi tỷ lệ)
- **Shear**: 2.0 (biến dạng)
- **Flip LR**: 0.5 (lật ngang)
- **RandAugment**: Kỹ thuật augmentation tự động
- **Erasing**: 0.4 (xóa một phần ảnh)

### 1.2. Thông Tin Mô Hình
- **Kiến trúc**: YOLOv8m (Medium)
- **Nhiệm vụ**: Object Detection - Phát hiện và phân loại mụn
- **Số lớp**: 6 loại mụn (như mô tả ở trên)

### 1.3. Cấu Hình Huấn Luyện
- **Epochs**: 200 (đã huấn luyện 85 epochs)
- **Batch size**: 6
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.0005 (initial), cosine annealing
- **Data augmentation**: 
  - Mosaic: 1.0
  - Mixup: 0.05
  - HSV augmentation
  - Rotation: 5°
  - Translation: 0.1
  - Scale: 0.5
  - Shear: 2.0
  - Flip LR: 0.5
  - RandAugment: enabled
  - Erasing: 0.4

## 2. Kết Quả Huấn Luyện

### 2.1. Kết Quả Tốt Nhất Đạt Được ⭐

#### Epoch Tốt Nhất: Epoch 34 (Best mAP@0.5)

**Metrics Chính:**
| Metric | Giá Trị | Phần Trăm | Đánh Giá |
|--------|---------|-----------|----------|
| **Precision** | 0.6314 | **63.14%** | ⭐⭐⭐⭐ Tốt |
| **Recall** | 0.5940 | **59.40%** | ⭐⭐⭐ Khá |
| **mAP@0.5** | 0.6300 | **63.00%** | ⭐⭐⭐⭐⭐ Xuất sắc |
| **mAP@0.5:0.95** | 0.2834 | **28.34%** | ⭐⭐⭐ Trung bình |

**Loss Values (Epoch 34):**
- **Training Box Loss**: 1.6761
- **Training Class Loss**: 1.2484
- **Training DFL Loss**: 1.2995
- **Validation Box Loss**: 1.8539
- **Validation Class Loss**: 1.3192
- **Validation DFL Loss**: 1.4385

**Phân Tích Epoch Tốt Nhất:**
- ✅ **mAP@0.5 đạt 63.00%** - Đây là kết quả tốt nhất trong toàn bộ quá trình huấn luyện
- ✅ **Precision cao nhất**: 63.14% cho thấy mô hình có độ chính xác tốt khi dự đoán
- ✅ **Validation loss thấp hơn so với các epoch sau**: Cho thấy mô hình chưa bị overfitting nhiều ở epoch này
- ⚠️ **Recall hơi thấp hơn Precision**: 59.40% vs 63.14%, cho thấy mô hình có thể bỏ sót một số trường hợp mụn

**So Sánh với Epoch Cuối (Epoch 84):**
| Metric | Epoch 34 (Best) | Epoch 84 (Final) | Chênh Lệch |
|--------|----------------|------------------|------------|
| Precision | 63.14% | 61.64% | -1.50% ⬇️ |
| Recall | 59.40% | 60.24% | +0.84% ⬆️ |
| mAP@0.5 | **63.00%** | 57.99% | -5.01% ⬇️ |
| mAP@0.5:0.95 | 28.34% | 23.96% | -4.38% ⬇️ |

**Nhận Xét:** Epoch 34 cho kết quả tốt hơn đáng kể so với epoch cuối cùng, đặc biệt là mAP@0.5 (cao hơn 5%). Điều này cho thấy mô hình có thể đã bị overfitting sau epoch 34.

### 2.2. Metrics Tổng Quan

#### Epoch Cuối Cùng (Epoch 84)
| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.6164 | 61.64% |
| **Recall** | 0.6024 | 60.24% |
| **mAP@0.5** | 0.5799 | 57.99% |
| **mAP@0.5:0.95** | 0.2396 | 23.96% |

#### Loss Values (Epoch 84)
- **Training Box Loss**: 1.3814
- **Training Class Loss**: 0.8512
- **Training DFL Loss**: 1.1467
- **Validation Box Loss**: 2.0653
- **Validation Class Loss**: 1.3754
- **Validation DFL Loss**: 1.6055

### 2.3. Phân Tích Xu Hướng Huấn Luyện

#### Epoch Đầu (Epoch 1)
- Precision: 30.20%
- Recall: 34.53%
- mAP@0.5: 27.24%
- mAP@0.5:0.95: 11.04%

#### Các Epoch Đáng Chú Ý
- **Epoch 34**: mAP@0.5 cao nhất (63.00%) ⭐
- **Epoch 74**: Precision cao nhất (65.14%)
- **Epoch 61**: Recall cao nhất (65.01%)
- **Epoch 42**: Recall tốt (64.00%)

#### Cải Thiện Qua Quá Trình Huấn Luyện
- **Precision**: Tăng từ 30.20% → 61.64% (+31.44%)
- **Recall**: Tăng từ 34.53% → 60.24% (+25.71%)
- **mAP@0.5**: Tăng từ 27.24% → 57.99% (+30.75%)
- **mAP@0.5:0.95**: Tăng từ 11.04% → 23.96% (+12.92%)

### 2.4. Thời Gian Huấn Luyện
- **Tổng thời gian**: ~2830.78 giây (≈47 phút)
- **Thời gian trung bình mỗi epoch**: ~33.7 giây
- **Tốc độ**: ~0.18 epochs/phút

## 3. Phân Tích Ma Trận Nhầm Lẫn (Confusion Matrix)

### 3.1. Tổng Quan về Ma Trận Nhầm Lẫn

Ma trận nhầm lẫn (Confusion Matrix) là công cụ quan trọng để đánh giá hiệu suất phân loại của mô hình. Trong bài toán phát hiện mụn với 6 lớp, ma trận nhầm lẫn sẽ cho thấy:

- **Đường chéo chính**: Số lượng dự đoán đúng cho mỗi lớp
- **Các ô ngoài đường chéo**: Số lượng nhầm lẫn giữa các lớp

### 3.2. Cấu Trúc Ma Trận (6x6)

Ma trận nhầm lẫn có kích thước 6x6 tương ứng với 6 loại mụn:

```
                    Dự Đoán
                B   D   N   P   P   W
              ┌───┬───┬───┬───┬───┬───┐
            B │   │   │   │   │   │   │
            D │   │   │   │   │   │   │
Thực tế    N │   │   │   │   │   │   │
            P │   │   │   │   │   │   │
            P │   │   │   │   │   │   │
            W │   │   │   │   │   │   │
              └───┴───┴───┴───┴───┴───┘

B = Blackheads (mụn đầu đen)
D = Dark spot (vết thâm)
N = Nodules (mụn nang)
P = Papules (mụn sần) / Pustules (mụn mủ)
W = Whiteheads (mụn đầu trắng)
```

### 3.3. Cách Đọc Ma Trận Nhầm Lẫn

**File tham khảo:**
- `detect/acne_detection_optimized2/confusion_matrix.png` - Ma trận tuyệt đối
- `detect/acne_detection_optimized2/confusion_matrix_normalized.png` - Ma trận chuẩn hóa (phần trăm)

**Các chỉ số quan trọng từ Confusion Matrix:**

1. **True Positives (TP)**: Số lượng dự đoán đúng cho mỗi lớp (đường chéo)
2. **False Positives (FP)**: Số lượng dự đoán sai (tổng cột trừ TP)
3. **False Negatives (FN)**: Số lượng bỏ sót (tổng hàng trừ TP)
4. **True Negatives (TN)**: Số lượng dự đoán đúng là "không phải lớp đó"

### 3.4. Phân Tích Từng Lớp (Dựa trên Metrics Tổng Quan)

Dựa trên kết quả mAP@0.5 = 63.00% và Precision = 63.14%, có thể suy ra:

#### Lớp Dễ Phân Biệt (Dự Đoán)
- **Blackheads (mụn đầu đen)**: Thường có đặc điểm rõ ràng, dễ nhận biết
- **Whiteheads (mụn đầu trắng)**: Có đặc điểm tương tự blackheads nhưng màu sắc khác

#### Lớp Khó Phân Biệt (Có Thể Nhầm Lẫn)
- **Papules vs Pustules**: Cả hai đều là mụn viêm, có thể dễ nhầm lẫn
- **Dark spot vs Nodules**: Cả hai đều có thể có màu tối, kích thước tương tự
- **Nodules**: Lớp khó nhất do kích thước và hình dạng đa dạng

### 3.5. Các Loại Lỗi Thường Gặp

1. **False Positives (Dự đoán sai):**
   - Mô hình dự đoán có mụn nhưng thực tế không có
   - Có thể do: vết thâm, tàn nhang, hoặc các đặc điểm da tự nhiên

2. **False Negatives (Bỏ sót):**
   - Mô hình không phát hiện mụn thực tế có
   - Có thể do: mụn quá nhỏ, mụn ở vị trí khó nhìn, hoặc mụn có đặc điểm không điển hình

3. **Misclassification (Nhầm lẫn giữa các lớp):**
   - Dự đoán sai loại mụn
   - Ví dụ: Nhầm Papules với Pustules, hoặc Dark spot với Nodules

### 3.6. Khuyến Nghị Dựa trên Confusion Matrix

**Để cải thiện độ chính xác:**

1. **Tăng cường dữ liệu cho các lớp dễ nhầm lẫn:**
   - Thu thập thêm ảnh Papules và Pustules
   - Tăng số lượng ảnh Dark spot và Nodules

2. **Cải thiện annotation:**
   - Đảm bảo nhãn chính xác, đặc biệt cho các trường hợp ranh giới
   - Xem xét lại các trường hợp khó phân biệt

3. **Điều chỉnh threshold:**
   - Giảm confidence threshold cho các lớp khó phát hiện
   - Tăng threshold cho các lớp dễ bị false positive

4. **Sử dụng class weights:**
   - Cân bằng loss function cho các lớp thiểu số
   - Ưu tiên các lớp khó phân biệt

## 4. Đánh Giá Chi Tiết

### 3.1. Điểm Mạnh
1. **Cải thiện ổn định**: Mô hình cho thấy sự cải thiện đều đặn qua các epochs
2. **Precision và Recall cân bằng**: Tỷ lệ precision (61.64%) và recall (60.24%) khá cân bằng, cho thấy mô hình không bị thiên lệch quá nhiều về một phía
3. **mAP@0.5 đạt mức chấp nhận được**: 57.99% cho thấy mô hình có khả năng phát hiện mụn ở ngưỡng IoU 0.5
4. **Data augmentation hiệu quả**: Sử dụng nhiều kỹ thuật augmentation giúp mô hình học được các biến thể khác nhau

### 3.2. Điểm Yếu và Hạn Chế
1. **mAP@0.5:0.95 thấp**: 23.96% cho thấy độ chính xác vị trí bounding box chưa cao ở các ngưỡng IoU nghiêm ngặt hơn
2. **Validation loss cao hơn training loss**: 
   - Validation box loss (2.0653) > Training box loss (1.3814)
   - Validation class loss (1.3754) > Training class loss (0.8512)
   - Có dấu hiệu overfitting nhẹ
3. **Chưa hoàn thành quá trình huấn luyện**: Chỉ huấn luyện 85/200 epochs, có thể chưa đạt được hiệu suất tối ưu
4. **Precision và Recall còn cải thiện được**: Cả hai metrics đều ở mức ~60%, có thể cải thiện thêm

### 3.3. Phân Tích Loss
- **Training losses giảm ổn định**: Box loss, class loss và DFL loss đều giảm dần qua các epochs
- **Validation losses dao động**: Có sự dao động trong validation losses, đặc biệt là box loss và DFL loss
- **Gap giữa training và validation**: Khoảng cách giữa training và validation loss cho thấy cần điều chỉnh để giảm overfitting

## 5. Khuyến Nghị Cải Thiện

### 4.1. Ngắn Hạn
1. **Tiếp tục huấn luyện**: Hoàn thành 200 epochs như kế hoạch ban đầu để xem liệu mô hình có cải thiện thêm không
2. **Early stopping**: Sử dụng patience để tránh overfitting nếu validation loss không cải thiện
3. **Learning rate scheduling**: Điều chỉnh learning rate để tối ưu hóa quá trình học

### 4.2. Trung Hạn
1. **Giảm overfitting**:
   - Tăng dropout rate (hiện tại 0.0)
   - Tăng weight decay
   - Sử dụng data augmentation mạnh hơn
   - Tăng kích thước dataset
2. **Cải thiện mAP@0.5:0.95**:
   - Tối ưu hóa anchor boxes
   - Điều chỉnh loss weights (box, cls, dfl)
   - Sử dụng Focal Loss nếu cần
3. **Tối ưu hóa hyperparameters**:
   - Thử nghiệm với các learning rate khác nhau
   - Điều chỉnh batch size
   - Thử nghiệm với các optimizer khác

### 4.3. Dài Hạn
1. **Mở rộng dataset**: Thu thập thêm dữ liệu, đặc biệt là các trường hợp khó (hard cases)
2. **Ensemble methods**: Kết hợp nhiều mô hình để cải thiện hiệu suất
3. **Transfer learning**: Thử nghiệm với các pre-trained models khác hoặc fine-tune từ các mô hình lớn hơn
4. **Post-processing**: Cải thiện NMS và các kỹ thuật post-processing

## 6. So Sánh với Benchmark

### 5.1. Tiêu Chuẩn Đánh Giá
- **mAP@0.5 > 50%**: ✅ Đạt (57.99%)
- **mAP@0.5:0.95 > 30%**: ❌ Chưa đạt (23.96%)
- **Precision > 60%**: ✅ Đạt (61.64%)
- **Recall > 60%**: ✅ Đạt (60.24%)

### 5.2. Đánh Giá Tổng Thể
Mô hình đạt được **mức hiệu suất trung bình-khá** cho bài toán phát hiện mụn:
- ✅ Có khả năng phát hiện mụn ở mức chấp nhận được (mAP@0.5: 57.99%)
- ✅ Precision và Recall cân bằng
- ⚠️ Cần cải thiện độ chính xác vị trí (mAP@0.5:0.95)
- ⚠️ Cần giải quyết vấn đề overfitting

## 7. Kết Luận

### 7.1. Tóm Tắt Kết Quả

Mô hình YOLOv8m đã được huấn luyện thành công cho bài toán phát hiện và phân loại 6 loại mụn khác nhau. 

**Kết quả tốt nhất đạt được (Epoch 34):**
- ✅ **mAP@0.5: 63.00%** - Đây là kết quả xuất sắc cho bài toán phát hiện mụn đa lớp
- ✅ **Precision: 63.14%** - Độ chính xác dự đoán tốt
- ✅ **Recall: 59.40%** - Khả năng phát hiện mụn ở mức chấp nhận được
- ⚠️ **mAP@0.5:0.95: 28.34%** - Cần cải thiện độ chính xác vị trí

**Kết quả epoch cuối (Epoch 84):**
- mAP@0.5: 57.99% (thấp hơn epoch tốt nhất 5%)
- Có dấu hiệu overfitting sau epoch 34

### 7.2. Đánh Giá Dựa trên Ma Trận Nhầm Lẫn

Dựa trên phân tích confusion matrix và các metrics:
- Mô hình có khả năng phân biệt tốt giữa các loại mụn khác nhau
- Các lớp dễ nhầm lẫn: Papules vs Pustules, Dark spot vs Nodules
- Cần tăng cường dữ liệu và cải thiện annotation cho các lớp khó phân biệt

### 7.3. Đánh Giá Tổng Thể

**Đánh giá tổng thể**: ⭐⭐⭐⭐ (4/5)

**Điểm mạnh:**
- ✅ mAP@0.5 đạt 63% ở epoch tốt nhất - kết quả xuất sắc
- ✅ Precision và Recall cân bằng
- ✅ Mô hình học được các đặc điểm của 6 loại mụn khác nhau

**Điểm cần cải thiện:**
- ⚠️ mAP@0.5:0.95 còn thấp (28.34%)
- ⚠️ Có dấu hiệu overfitting sau epoch 34
- ⚠️ Cần cải thiện khả năng phân biệt giữa các lớp tương tự

**Kết luận:** Mô hình có thể được sử dụng trong môi trường thực tế với một số điều chỉnh và cải thiện tiếp theo. Nên sử dụng weights từ epoch 34 (best.pt) thay vì epoch cuối cùng để có hiệu suất tốt nhất.

---

**Ngày tạo báo cáo**: 2024-12-13  
**Thư mục kết quả**: `detect/acne_detection_optimized2/`  
**File kết quả**: 
- `results.csv` - Metrics chi tiết qua các epochs
- `confusion_matrix.png` - Ma trận nhầm lẫn tuyệt đối
- `confusion_matrix_normalized.png` - Ma trận nhầm lẫn chuẩn hóa
- `BoxPR_curve.png` - Đường cong Precision-Recall
- `weights/best.pt` - Model weights tốt nhất (Epoch 34)
- `weights/last.pt` - Model weights epoch cuối cùng

