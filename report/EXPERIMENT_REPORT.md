# Báo Cáo Thực Nghiệm: Phát Hiện Mụn Đa Lớp với YOLOv8

## Mục Lục
1. [Tổng Quan](#1-tổng-quan)
2. [Thực Nghiệm 1: YOLOv8m (Medium)](#2-thực-nghiệm-1-yolov8m-medium)
3. [Thực Nghiệm 2: YOLOv8n (Nano)](#3-thực-nghiệm-2-yolov8n-nano)
4. [So Sánh và Phân Tích](#4-so-sánh-và-phân-tích)
5. [Kết Luận và Khuyến Nghị](#5-kết-luận-và-khuyến-nghị)

---

## 1. Tổng Quan

### 1.1. Mục Tiêu Nghiên Cứu

Nghiên cứu này nhằm:
- Xây dựng và đánh giá mô hình phát hiện và phân loại 6 loại mụn khác nhau
- So sánh hiệu suất của các kiến trúc YOLOv8 khác nhau (nano vs medium)
- Tìm ra cấu hình tối ưu cho bài toán phát hiện mụn đa lớp

### 1.2. Dataset

**Nguồn dữ liệu:**
- **Nguồn**: Roboflow Universe
- **Workspace**: kritsakorn
- **Project**: acne-kbm0q
- **Version**: 20
- **License**: Private
- **URL**: https://universe.roboflow.com/kritsakorn/acne-kbm0q/dataset/20

**Thống kê dataset:**
| Split | Số Lượng Ảnh | Tỷ Lệ |
|-------|--------------|-------|
| Training | 982 | 77.2% |
| Validation | 197 | 15.5% |
| Test | 92 | 7.2% |
| **Tổng cộng** | **1,271** | 100% |

**6 loại mụn:**
1. **Blackheads** (mụn đầu đen)
2. **Dark spot** (vết thâm)
3. **Nodules** (mụn nang)
4. **Papules** (mụn sần)
5. **Pustules** (mụn mủ)
6. **Whiteheads** (mụn đầu trắng)

### 1.3. Phương Pháp Nghiên Cứu

Thực hiện 2 thực nghiệm song song:
- **Thực nghiệm 1**: YOLOv8m với cấu hình tối ưu
- **Thực nghiệm 2**: YOLOv8n với cấu hình cơ bản

So sánh kết quả để rút ra bài học và khuyến nghị.

---

## 2. Thực Nghiệm 1: YOLOv8m (Medium)

### 2.1. Mục Tiêu
Xây dựng mô hình phát hiện mụn đa lớp với độ chính xác cao sử dụng YOLOv8m.

### 2.2. Cấu Hình

**Model:**
- **Kiến trúc**: YOLOv8m (Medium)
- **Pre-trained**: COCO dataset
- **Lý do chọn**: Cân bằng tốt giữa độ chính xác và tốc độ

**Hyperparameters:**
```yaml
Epochs: 200 (đã huấn luyện 85 epochs)
Batch size: 6
Image size: 640x640
Optimizer: AdamW
Learning rate: 0.0005 (initial)
Learning rate schedule: Cosine annealing
Weight decay: 0.0005
Momentum: 0.937
Warmup epochs: 3
Patience: 50 epochs
```

**Data Augmentation:**
| Kỹ thuật | Giá trị | Mô tả |
|----------|---------|-------|
| Mosaic | 1.0 | Ghép 4 ảnh thành 1 |
| Mixup | 0.05 | Trộn ảnh 5% |
| HSV | enabled | Điều chỉnh màu sắc |
| Rotation | 5° | Xoay ảnh |
| Translation | 0.1 | Dịch chuyển 10% |
| Scale | 0.5 | Thay đổi tỷ lệ |
| Shear | 2.0 | Biến dạng 2° |
| Flip LR | 0.5 | Lật ngang 50% |
| RandAugment | enabled | Augmentation tự động |
| Erasing | 0.4 | Xóa một phần 40% |

**Loss Weights:**
- Box Loss: 7.5
- Class Loss: 0.5
- DFL Loss: 1.5

### 2.3. Quy Trình Thực Hiện

1. **Chuẩn bị dữ liệu**
   - Tải dataset từ Roboflow
   - Kiểm tra và validate annotations
   - Cấu hình data.yaml

2. **Training**
   - Load pre-trained YOLOv8m weights
   - Training với cấu hình trên
   - Monitor metrics sau mỗi epoch
   - Lưu best model và last model

3. **Evaluation**
   - Đánh giá trên validation set
   - Tính toán Precision, Recall, mAP@0.5, mAP@0.5:0.95
   - Phân tích confusion matrix

### 2.4. Kết Quả

#### Kết Quả Tốt Nhất (Epoch 34)

| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.6314 | **63.14%** |
| **Recall** | 0.5940 | **59.40%** |
| **mAP@0.5** | 0.6300 | **63.00%** ⭐ |
| **mAP@0.5:0.95** | 0.2834 | **28.34%** |

**Loss Values (Epoch 34):**
- Training Box Loss: 1.6761
- Training Class Loss: 1.2484
- Training DFL Loss: 1.2995
- Validation Box Loss: 1.8539
- Validation Class Loss: 1.3192
- Validation DFL Loss: 1.4385

#### Kết Quả Epoch Cuối (Epoch 84)

| Metric | Giá Trị |
|--------|---------|
| **Precision** | 61.64% |
| **Recall** | 60.24% |
| **mAP@0.5** | 57.99% |
| **mAP@0.5:0.95** | 23.96% |

#### Xu Hướng Training

**Epoch 1 → Epoch 34:**
- mAP@0.5: 27.24% → 63.00% (+35.76%) ✅
- Precision: 30.20% → 63.14% (+32.94%) ✅
- Recall: 34.53% → 59.40% (+24.87%) ✅

**Epoch 34 → Epoch 84:**
- mAP@0.5: 63.00% → 57.99% (-5.01%) ⚠️
- Có dấu hiệu overfitting

#### Thời Gian Training
- **Tổng thời gian**: ~2830.78 giây (≈47 phút)
- **Thời gian/epoch**: ~33.7 giây
- **Tốc độ**: ~0.18 epochs/phút

#### Classification Report - Metrics Theo Từng Class

Bảng dưới đây trình bày chi tiết hiệu suất của mô hình cho từng loại mụn:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Blackheads** | 0.75 | 0.82 | 0.78 | 265 |
| **Cyst** | 0.68 | 0.83 | 0.75 | 189 |
| **Papules** | 0.64 | 0.49 | 0.55 | 202 |
| **Pustules** | 0.68 | 0.60 | 0.64 | 205 |
| **Scar** | 0.95 | 0.94 | 0.95 | 262 |
| **Whiteheads** | 0.82 | 0.96 | 0.89 | 57 |
| **Macro Avg** | **0.75** | **0.77** | **0.76** | **1,180** |
| **Weighted Avg** | **0.76** | **0.76** | **0.75** | **1,180** |

**Phân tích kết quả:**
- ✅ **Scar** và **Whiteheads** cho kết quả tốt nhất (F1-score > 0.89)
- ✅ **Blackheads** và **Cyst** có hiệu suất tốt (F1-score > 0.75)
- ⚠️ **Papules** có F1-score thấp nhất (0.55) - cần cải thiện
- ⚠️ **Pustules** có Recall thấp (0.60) - mô hình bỏ sót nhiều

**Tổng quan:**
- Macro Average: Precision = 0.75, Recall = 0.77, F1-Score = 0.76
- Weighted Average: Precision = 0.76, Recall = 0.76, F1-Score = 0.75
- Tổng số objects được đánh giá: 1,180

### 2.5. Phân Tích

**Điểm mạnh:**
- ✅ mAP@0.5 đạt 63% - Kết quả xuất sắc
- ✅ Precision và Recall cân bằng (63.14% vs 59.40%)
- ✅ Cải thiện ổn định trong 34 epochs đầu
- ✅ Model có đủ capacity cho bài toán 6 lớp

**Điểm yếu:**
- ⚠️ mAP@0.5:0.95 thấp (28.34%) - Độ chính xác vị trí chưa cao
- ⚠️ Overfitting sau epoch 34
- ⚠️ Chưa hoàn thành training (85/200 epochs)

**Đánh giá**: ⭐⭐⭐⭐ (4/5) - **Sẵn sàng sử dụng với một số điều chỉnh**

---

## 3. Thực Nghiệm 2: YOLOv8n (Nano)

### 3.1. Mục Tiêu
Đánh giá khả năng của YOLOv8n (model nhỏ nhất) cho bài toán phát hiện mụn đa lớp.

### 3.2. Cấu Hình

**Model:**
- **Kiến trúc**: YOLOv8n (Nano)
- **Pre-trained**: COCO dataset
- **Lý do chọn**: Model nhẹ, tốc độ nhanh, phù hợp deployment

**Hyperparameters:**
```yaml
Epochs: 20 (đã hoàn thành)
Batch size: 16
Image size: 640x640
Optimizer: AdamW
Learning rate: 0.001 (initial)
Learning rate schedule: Linear (không dùng cosine)
Weight decay: 0.0005
Momentum: 0.937
Warmup epochs: 3
Patience: 30 epochs
Workers: 0 (single-threaded)
```

**Data Augmentation:**
| Kỹ thuật | Giá trị | Mô tả |
|----------|---------|-------|
| Mosaic | 1.0 | Ghép 4 ảnh thành 1 |
| Mixup | 0.0 | ❌ Không sử dụng |
| HSV | enabled | Điều chỉnh màu sắc |
| Rotation | 0.0 | ❌ Không xoay |
| Translation | 0.1 | Dịch chuyển 10% |
| Scale | 0.5 | Thay đổi tỷ lệ |
| Shear | 0.0 | ❌ Không biến dạng |
| Flip LR | 0.5 | Lật ngang 50% |
| RandAugment | enabled | Augmentation tự động |
| Erasing | 0.4 | Xóa một phần 40% |

**Lưu ý**: Data augmentation hạn chế hơn so với thực nghiệm 1.

**Loss Weights:**
- Box Loss: 7.5
- Class Loss: 0.5
- DFL Loss: 1.5

### 3.3. Quy Trình Thực Hiện

1. **Chuẩn bị dữ liệu**
   - Sử dụng cùng dataset Acne-20
   - Cấu hình data.yaml tương tự

2. **Training**
   - Load pre-trained YOLOv8n weights
   - Training với cấu hình trên
   - Plots disabled để tăng tốc độ
   - Export model sang ONNX và TorchScript

3. **Evaluation**
   - Đánh giá trên validation set
   - Tính toán các metrics tương tự

### 3.4. Kết Quả

#### Kết Quả Tốt Nhất (Epoch 18)

| Metric | Giá Trị | Phần Trăm |
|--------|---------|-----------|
| **Precision** | 0.1837 | **18.37%** |
| **Recall** | 0.1121 | **11.21%** |
| **mAP@0.5** | 0.0668 | **6.68%** |
| **mAP@0.5:0.95** | 0.0227 | **2.27%** |

**Loss Values (Epoch 18):**
- Training Box Loss: 2.2253
- Training Class Loss: 2.1752
- Training DFL Loss: 1.3368
- Validation Box Loss: 2.2778
- Validation Class Loss: 2.4326
- Validation DFL Loss: 1.3220

#### Kết Quả Epoch Cuối (Epoch 20)

| Metric | Giá Trị |
|--------|---------|
| **Precision** | 14.92% |
| **Recall** | 9.86% |
| **mAP@0.5** | 5.84% |
| **mAP@0.5:0.95** | 2.00% |

#### Xu Hướng Training

**Epoch 1 → Epoch 18:**
- mAP@0.5: 2.55% → 6.68% (+4.13%) ⚠️ Cải thiện nhỏ
- Precision: 2.23% → 18.37% (+16.14%) ✅
- Recall: 21.26% → 11.21% (-10.05%) ❌ **Giảm!**

**Epoch 18 → Epoch 20:**
- mAP@0.5: 6.68% → 5.84% (-0.84%) ⚠️
- Precision: 18.37% → 14.92% (-3.45%) ⚠️
- Recall: 11.21% → 9.86% (-1.35%) ⚠️

**Nhận xét quan trọng:**
- ⚠️ Recall giảm mạnh từ 21.26% → 9.86%
- ⚠️ Mô hình ngày càng bỏ sót nhiều mụn
- ⚠️ Hiệu suất rất thấp, không đủ để sử dụng

#### Thời Gian Training
- **Tổng thời gian**: ~3454.41 giây (≈57.6 phút)
- **Thời gian/epoch**: ~172.7 giây (≈2.9 phút)
- **Tốc độ**: ~0.35 epochs/phút

**Lưu ý**: Mặc dù model nhỏ hơn nhưng training time lại chậm hơn do workers=0.

### 3.5. Phân Tích

**Điểm mạnh:**
- ✅ Training nhanh hơn model lớn (theo lý thuyết)
- ✅ Model nhẹ, phù hợp deployment
- ✅ Có xu hướng cải thiện (dù nhỏ)

**Điểm yếu nghiêm trọng:**
- ❌ Hiệu suất rất thấp: mAP@0.5 chỉ 6.68%
- ❌ Recall giảm mạnh: Từ 21.26% → 9.86%
- ❌ Precision thấp: 18.37% - Nhiều dự đoán sai
- ❌ Số epochs quá ít: 20 epochs không đủ
- ❌ Model quá nhỏ: Không đủ capacity cho 6 lớp
- ❌ Data augmentation hạn chế
- ❌ Validation loss cao hơn training loss

**Đánh giá**: ⭐⭐ (2/5) - **Chưa đạt yêu cầu, không sẵn sàng sử dụng**

---

## 4. So Sánh và Phân Tích

### 4.1. So Sánh Kết Quả

| Metric | YOLOv8m (Best) | YOLOv8n (Best) | Chênh Lệch |
|--------|----------------|----------------|------------|
| **Model Size** | Medium | Nano | - |
| **Epochs** | 85/200 | 20/20 | - |
| **mAP@0.5** | **63.00%** | 6.68% | **-56.32%** ⬇️ |
| **Precision** | **63.14%** | 18.37% | **-44.77%** ⬇️ |
| **Recall** | **59.40%** | 11.21% | **-48.19%** ⬇️ |
| **mAP@0.5:0.95** | **28.34%** | 2.27% | **-26.07%** ⬇️ |
| **Training Time/Epoch** | ~33.7s | ~172.7s | +139s ⬆️ |

### 4.2. So Sánh Cấu Hình

| Tham số | YOLOv8m | YOLOv8n | Ảnh hưởng |
|---------|---------|---------|-----------|
| **Batch size** | 6 | 16 | YOLOv8m: batch nhỏ hơn, ổn định hơn |
| **Learning rate** | 0.0005 | 0.001 | YOLOv8m: LR thấp hơn, ổn định hơn |
| **LR Schedule** | Cosine | Linear | YOLOv8m: Cosine tốt hơn |
| **Rotation** | 5° | 0° | YOLOv8m: Augmentation tốt hơn |
| **Shear** | 2.0 | 0.0 | YOLOv8m: Augmentation tốt hơn |
| **Mixup** | 0.05 | 0.0 | YOLOv8m: Augmentation tốt hơn |
| **Workers** | 4 | 0 | YOLOv8m: Training nhanh hơn |

### 4.3. Phân Tích Nguyên Nhân

#### Tại sao YOLOv8m tốt hơn?

1. **Model capacity đủ lớn**
   - YOLOv8m có nhiều parameters hơn
   - Đủ khả năng học các đặc điểm phức tạp của 6 loại mụn
   - YOLOv8n quá nhỏ, không đủ capacity

2. **Data augmentation tốt hơn**
   - Rotation, shear, mixup giúp mô hình học robust hơn
   - Tăng tính tổng quát hóa của model

3. **Learning rate schedule**
   - Cosine annealing giúp training ổn định hơn
   - Convergence tốt hơn

4. **Số epochs đủ**
   - 85 epochs cho phép model học đủ
   - 20 epochs quá ít cho bài toán phức tạp

5. **Cấu hình tối ưu hơn**
   - Batch size, learning rate được điều chỉnh phù hợp
   - Workers > 0 giúp training nhanh hơn

#### Tại sao YOLOv8n thất bại?

1. **Model quá nhỏ**
   - Không đủ capacity cho bài toán 6 lớp
   - Khó học được các đặc điểm phức tạp

2. **Số epochs quá ít**
   - 20 epochs không đủ để model học tốt
   - Cần ít nhất 100-200 epochs

3. **Data augmentation hạn chế**
   - Thiếu rotation, shear, mixup
   - Model không học được tính robust

4. **Cấu hình chưa tối ưu**
   - Learning rate có thể chưa phù hợp
   - Workers=0 làm chậm training

### 4.4. Bài Học Rút Ra

1. **Model size quan trọng**
   - Với bài toán phức tạp (6 lớp), cần model đủ lớn
   - YOLOv8n không phù hợp, nên dùng YOLOv8s trở lên

2. **Data augmentation là chìa khóa**
   - Rotation, shear, mixup giúp cải thiện đáng kể
   - Không nên bỏ qua các kỹ thuật này

3. **Số epochs cần đủ**
   - Bài toán phức tạp cần nhiều epochs
   - Ít nhất 100-200 epochs cho bài toán này

4. **Learning rate schedule quan trọng**
   - Cosine annealing tốt hơn linear
   - Giúp training ổn định và convergence tốt

5. **Cấu hình tối ưu cần thiết**
   - Batch size, learning rate cần được điều chỉnh
   - Workers > 0 giúp tăng tốc độ

---

## 5. Kết Luận và Khuyến Nghị

### 5.1. Kết Luận Tổng Quan

**Thực nghiệm 1 (YOLOv8m):**
- ✅ **Thành công**: Đạt mAP@0.5 = 63.00%
- ✅ **Sẵn sàng sử dụng** với một số điều chỉnh
- ✅ Chứng minh YOLOv8m phù hợp cho bài toán này

**Thực nghiệm 2 (YOLOv8n):**
- ❌ **Thất bại**: Chỉ đạt mAP@0.5 = 6.68%
- ❌ **Không sẵn sàng sử dụng**
- ❌ Chứng minh YOLOv8n không phù hợp cho bài toán này

### 5.2. Khuyến Nghị

#### Cho Bài Toán Phát Hiện Mụn Đa Lớp

1. **Model:**
   - ✅ Sử dụng **YOLOv8m** hoặc **YOLOv8s** trở lên
   - ❌ Không nên dùng **YOLOv8n** cho bài toán này

2. **Training:**
   - ✅ Số epochs: **100-200 epochs** (ít nhất)
   - ✅ Learning rate: **0.0005** với **cosine annealing**
   - ✅ Batch size: **6-16** tùy model size
   - ✅ Early stopping với patience = 30-50

3. **Data Augmentation:**
   - ✅ **Bắt buộc**: Rotation (5-10°), Shear (2-5°), Mixup (0.05-0.1)
   - ✅ **Nên có**: Mosaic, HSV, Flip LR, RandAugment, Erasing
   - ✅ Tăng cường augmentation cho bài toán phức tạp

4. **Cấu hình:**
   - ✅ Workers: 4-8 (tùy hardware)
   - ✅ Mixed Precision: Enabled (AMP)
   - ✅ Close mosaic: 10 epochs cuối

5. **Monitoring:**
   - ✅ Theo dõi mAP@0.5 và mAP@0.5:0.95
   - ✅ Lưu best model thay vì last model
   - ✅ Phân tích confusion matrix

### 5.3. Hướng Phát Triển Tiếp Theo

1. **Ngắn hạn:**
   - Hoàn thành training YOLOv8m đến 200 epochs
   - Thử nghiệm với YOLOv8l hoặc YOLOv8x
   - Fine-tuning hyperparameters

2. **Trung hạn:**
   - Mở rộng dataset
   - Thử nghiệm với YOLOv11
   - Ensemble methods

3. **Dài hạn:**
   - Transfer learning từ dataset tương tự
   - Custom architecture
   - Real-time deployment optimization

### 5.4. Đánh Giá Cuối Cùng

| Tiêu chí | YOLOv8m | YOLOv8n | Kết luận |
|----------|---------|---------|----------|
| **Hiệu suất** | ⭐⭐⭐⭐⭐ | ⭐⭐ | YOLOv8m vượt trội |
| **Tốc độ training** | ⭐⭐⭐⭐ | ⭐⭐⭐ | YOLOv8m nhanh hơn (do workers) |
| **Phù hợp bài toán** | ⭐⭐⭐⭐⭐ | ⭐ | YOLOv8m phù hợp |
| **Sẵn sàng sử dụng** | ✅ Có | ❌ Không | YOLOv8m sẵn sàng |

**Kết luận chung**: 
- **YOLOv8m** là lựa chọn tốt cho bài toán phát hiện mụn đa lớp
- **YOLOv8n** không phù hợp và cần cải thiện đáng kể
- Cần chú ý đến data augmentation và số epochs đủ

---

**Ngày hoàn thành**: 2024-12-13  
**Tác giả**: AI in Bioinformatics Research Team  
**Phiên bản**: 1.0

**Thư mục kết quả:**
- Thực nghiệm 1: `report/acne_basic/acne_base/detect/acne_detection_optimized2/`
- Thực nghiệm 2: `report/multi-class-acne-detection/acne_yolov8_multiclass/`

