# Phân loại Tin nhắn Spam bằng phương pháp Naive Bayes 

Dự án tập trung vào bài toán Text Classification (Phân loại văn bản). Mục tiêu là xây dựng một chương trình khả năng phân loại tin nhắn là **Spam** (tin rác) hay **Ham** (tin thường).

Phương pháp tiếp cận chính:

**Naive Bayes Classifier:** Một thuật toán học máy truyền thống hiệu quả cho phân loại văn bản.

## Các Luồng Xử Lý (Pipelines)

**1. Tiền xử lý dữ liệu (Preprocessing)**
Mỗi tin nhắn thô được làm sạch qua các bước nghiêm ngặt để chuẩn hóa dữ liệu đầu vào.

  * **Lowercase:** Chuyển về chữ thường.
  * **Punctuation Removal:** Xóa bỏ dấu câu.
  * **Tokenize:** Tách câu thành các từ (tokens).
  * **Remove Stopword:** Loại bỏ các từ dừng (như 'the', 'is', 'are').
  * **Stemming:** Đưa từ về dạng gốc (ví dụ: 'studying' -\> 'studi').

**2. Tạo đặc trưng (Feature Creation - Bag-of-Words)**
Sau khi tiền xử lý, chúng ta xây dựng một bộ từ điển (Dictionary) chứa tất cả các từ duy nhất trong tập dữ liệu. Mỗi tin nhắn sau đó được biểu diễn dưới dạng một vector, đếm tần suất xuất hiện của mỗi từ trong từ điển (phương pháp Bag-of-Words).

**3. Huấn luyện mô hình**
Vector đặc trưng (X) và nhãn (y) được đưa vào mô hình **Gaussian Naive Bayes**. Mô hình này học xác suất của mỗi từ xuất hiện trong các lớp "Spam" và "Ham", dựa trên định lý Bayes.

Công thức Naive Bayes: $P(A|B)=\frac{P(B|A).P(A)}{P(B)}$ 

-----

## Công nghệ sử dụng

  * **Xử lý dữ liệu:** Pandas, NumPy.
  * **Xử lý Ngôn ngữ Tự nhiên (NLP):** NLTK.
  * **Học máy (ML):** Scikit-learn (cho `GaussianNB`, `train_test_split`).

-----

## Kết quả

Dưới đây là kết quả độ chính xác (Accuracy) của hai phương pháp trên tập kiểm thử (Test set).

| Cấu hình | Độ chính xác (Test Accuracy) |
| :--- | :--- |
| GaussianNB | **86.02%** |

*Kết quả cho thấy mô hình Naive Bayes phân loại đúng khoảng 86.02% tin nhắn là spam hoặc không spam. Đây là một kết quả khá tốt, nhưng cũng có thể có những điểm cần cải thiện, như thử các thuật toán khác hoặc tinh chỉnh thêm các tham số của mô hình.*

-----

## Cài đặt và Chạy thử

Bạn có thể clone repository này và chạy thử dự án trên môi trường của mình (ví dụ: Google Colab).

```bash
# 1. Tải repository
git clone github.com/DuTi2201/spam-classification.git
cd spam-classification

# 2. Chạy file notebook.
```

**Lưu ý:** Cần tải tài nguyên NLTK:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
