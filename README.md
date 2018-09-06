# Mô hình dự đoán giọng nói vùng miền 

## 1. Cấu trúc thư mục 

Thư mục bao gồm các file/thư mục con: 

* `preprocess.py`: convert các file về `.wav`, sample rate = 16000, mono.

* `train.py`: train model.

* `inference.py`: dự đoán các files mới trong một thư mục, ví dụ `./data/private_test/`.

* `split.py`: phân chia dữ liệu cho _tập huấn luyện_ (training set) và _tập xác thực_ (validation set).

* `data.py`: tạo vector đặc trưng và DataLoader.

* `nets.py`: định nghĩa các mạng neuron.

* `predicts.py`: các hàm phục vụ việc dự đoán.

* `utils.py`: các hàm phụ trợ khác.

* `config.py`: các thông số mô hình và đường dẫn tới các thư mục chứa dữ liệu.

* `./saved_model`: thư mục chứa các model.

* `./csv_data`: thư mục chứa các file csv về đường dẫn tới file và nhãn tương ứng:
  * `training_groundtruth`: dữ liệu train.

  * `test_groundtruth`: dữ liệu public test.

## 2. Dự đoán các file trong một thư mục mới

Nếu chỉ muốn dự đoán nhãn của các file trong thư mục mới, giả sử `./data/private_test/`, các bước thực hiện như sau: 

1. Mở file `config.py` thay đổi các biến:

```python
BASE_ORIGINAL_PRIVATE_TEST = './data/private_test/'
BASE_PRIVATE_TEST = './data/wav' + RATE + '/private_test/'
INFER_ONLY = True
```
trong đó, `BASE_ORIGINAL_PRIVATE_TEST` là đường dẫn tới thư mục đó và `BASE_PRIVATE_TEST` là đường dẫn tới thư mục chứa các file âm thanh đã được convert sang sample rate 16000, mono. `INFER_ONLY = True` để chỉ convert các file cần dự đoán. Biến này cần được gán bằng `False` nếu muốn huấn luyện lại từ đầu. 

1. Chạy `python preprocess.py` để convert các file ra wav, 16000, mono. 

2. Chạy `python inference.py`. Sau khi chạy xong file này, kết quả sẽ được lưu vào `./result/submission.csv`

## 3. Huấn luyện mô hình 
Nếu muốn huấn luyện mô hình lại từ đầu, cần làm theo các bước:

1. Mở file `config.py`, sửa các dòng:

```python
BASE_ORIGINAL_TRAIN = './data/train/'
BASE_ORIGINAL_PUBLIC_TEST = './data/public_test/'
INFER_ONLY = False
```

trong đó `BASE_ORIGINAL_TRAIN` là đường dẫn tới thư mục chứa training file, `BASE_ORIGINAL_PUBLIC_TEST` là đường dẫn tới các thư mục chứa public test file. 

2. Chạy `python preprocess.py`

3. Chạy:

```bash
python train.py -r 3
python train.py -r 5
python train.py -r 7
python train.py -r 9
python train.py -r 77
```

trong đó `-r 3` để tạo các `random_state` khác nhau cho các hàm sinh ngẫu nhiên. Các số `3, 5, 7, 9, 77` có thể là các số `int` bất kỳ, miễn là chúng khác nhau. Các tham số mô hình khác có thể được điều chỉnh trong file `train.py`.

1. Sau khi chạy xong, các model sẽ được lưu trong `saved_model`.
2. Nếu muốn sử dụng các model mới được train này để dự đoán, ta cần đưa các đường dẫn của các model vào biến `model_path_fns` trong file `inference.py`

## 4. Hướng giải quyết bài toán

1. Chuẩn hóa các file bằng cách đưa chúng về cùng waveform 16000, mono.

2. Với mỗi file, một đoạn dài 1.5 giây được cắt ra ngẫu nhiên để tạo đặc trưng. Nếu file ngắn hơn 1.5 giây, ta bù thêm các giá trị bằng 0 về hai phía. Số giá trị bằng 0 ở mỗi phía là ngẫu nhiên sao cho tổng độ dài của đoạn là 1.5 giây. Con số 1.5 có thể được thay đổi qua biến `duration` trong file `train.py`. Các thí nghiệm cho thấy 1.5 cho kết quả khá tốt.

3. Với mỗi đoạn 1.5 giây được cắt ra, tạo đặc trưng `log_specgram` như trong file `data.py`. Đặc trưng này là một ma trận hai chiều. Ma trận này được resize về kích thước `224x224` rồi lặp lại ba lần để được một mảng ba chiều `3x224x224`. Mảng ba chiều này được coi như một bức ảnh màu để đưa vào mạng ResNet18. Mỗi bức ảnh này được gán nhãn của file audio mà đoạn 1.5 giây được cắt ra.

4. Khi dự đoán một file âm thanh mới, ta cắt ngẫu nhiên ra nhiều đoạn 1.5 giây khác nhau. Dùng các mô hình đã được huấn luyện để dự đoán từng đoạn. Lấy tổng các `score` (trước khi thực hiện softmax để tìm xác suất). Lớp tương ứng với `score` cao nhất sẽ tương ứng với nhãn của file âm thanh.

**Star nếu bạn thấy repo hữu ích :-)**
-- Tiep Vu --