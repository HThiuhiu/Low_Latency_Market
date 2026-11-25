[English](README.md) | **Tiếng Việt**

# Từ pipeline dữ liệu thị trường đến thuật toán giao dịch thích nghi

Repo này đi theo một con đường gồm 5 chương. Bắt đầu bằng việc lọc và tính toán trên dữ liệu thị trường thô. Tiếp
theo là xây và thử nghiệm mô hình trên dữ liệu đó. Cuối cùng là một thuật toán giao dịch thích nghi theo thời gian,
cùng một engine nhanh để nó tìm kiếm tốt. Mỗi chương là một nhánh, có pull request nối lên nhánh trước.

> Chỉ giao dịch giấy. Không có lệnh thật nào được gửi đi. Đây không phải lời khuyên đầu tư.

## Câu chuyện

### 1. Lọc và tính toán trên dữ liệu: pipeline xử lý dữ liệu

Câu hỏi đầu tiên: làm sao lấy dữ liệu thô của sàn, làm sạch, và tính toán trên nó mà không phí thời gian.

- **Pipeline realtime** (code gốc của repo): dữ liệu Binance trực tiếp, feature tính tăng dần, ring buffer,
  microservice qua Redis Streams. Có đo và tối ưu CPU, I/O, xem
  [docs/LOW_LATENCY_PIPELINE.vi.md](docs/LOW_LATENCY_PIPELINE.vi.md).
- **Cube ngày đúng thời điểm** (`step-01/data-pipeline`): mọi perp USDT từng niêm yết, giữ cả coin đã delist, lọc
  coin chết, universe top-40 mỗi tháng chỉ xếp hạng bằng dữ liệu quá khứ, funding khớp theo ngày.
- **Tín hiệu tính từ cube** (`step-02/signal-features`): 17 tín hiệu ngày từ các bài nghiên cứu, xếp hạng giữa các
  coin mỗi ngày.

### 2. Xây mô hình trên dữ liệu

- **Mô hình đầu tiên** (trong pipeline): CNN-BiLSTM-attention dự báo trên nến phút, chạy bằng ONNX Runtime.
- **Mô hình book xếp hạng chéo** (`step-03/book-model`): tín hiệu thành điểm mỗi ngày. Điểm thành book long/short:
  chân theo nghịch đảo biến động, hedge beta, chặn short bị squeeze, có phí và funding. Backtest kiểu sàn chạy lại
  book trên tài khoản USDT.

### 3. Thử nghiệm các mô hình

Mô hình nến phút không có lợi thế sau khi trừ phí. Trước khi tin mô hình tiếp theo, dự án cần công cụ biết nói
"không".

- **Bộ công cụ nghiên cứu** (`step-04/research-toolkit`): walk-forward có purge, CPCV, Sharpe giảm phát, xác suất
  backtest quá khớp (PBO), bootstrap theo cặp, sàng 6 khối thời gian, vault niêm phong cho test sạch.
- **C5** (`step-05/strategy-c5`): book rút ra sau các lần thử nghiệm. Điểm 6 tín hiệu, chân long hẹp, chân short
  rộng, bỏ short có funding âm, hedge beta. Xem [docs/C5.md](docs/C5.md).

### 4. Thuật toán thích nghi theo thời gian

C5 giữ trọng số tín hiệu cố định. Thị trường thay đổi, nên câu hỏi tiếp theo: nếu mỗi ngày chọn lại trọng số thì sao?

- **Trọng số thích nghi** (`step-06/adaptive-weights`): mỗi ngày tìm trong không gian trọng số bộ có mean / std 30
  ngày qua tốt nhất, tinh chỉnh quanh nó, rồi giao dịch ngày hôm sau bằng bộ đó. Không nhìn trước: test tính lại
  từng ngày trên cube kết thúc đúng ngày đó.
- **Triển khai** (`step-07/paper-dashboard`): bot paper chạy song song C5 và book thích nghi trên sổ lệnh thật, chạy
  bù những ngày máy tắt, hiện cả hai trên dashboard chạy trên máy.

Trong các test walk-forward, trọng số thích nghi chưa thắng trọng số cố định. Nó chạy như một book nghiên cứu có
theo dõi. Xem [docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md).

### 5. Tối ưu tính toán và tốc độ cho thuật toán thích nghi

Bản thích nghi đầu tiên chạy từng bộ trọng số qua book, lần lượt từng bộ. Kết quả đúng, nhưng phải giữ bộ tìm nhỏ, nên
chỉ phủ không gian trọng số rất thưa.

- **Engine chạy theo lô** (`step-08/fast-adaptive-engine`): bộ trọng số thành một chiều của tensor. Một lượt qua các
  ngày xếp hạng coin cho mọi bộ cùng lúc, giữ mọi book con trong một tensor. Chạy trên CPU hoặc GPU.
- **Bớt tính lại**: bộ phủ chạy một lần cho cả giai đoạn, điểm cửa sổ trượt lấy từ tổng cộng dồn, mỗi ngày chỉ mô
  phỏng lại nhóm nhỏ ứng viên tinh chỉnh.
- **Kiểm với bản gốc**: giữ book chậm làm bản tham chiếu, test kiểm hai cách cho cùng lãi lỗ từng ngày.

Giờ việc tìm kiếm phủ toàn bộ không gian trọng số mỗi ngày, và đoạn dài chạy được trên GPU.

## Bản đồ code

| Chương | Module | Làm gì |
|---|---|---|
| 1 | `qforecast/services/`, `core/`, `storage/` | Nhận dữ liệu realtime, feature tăng dần, Redis Streams / bus trong RAM |
| 1 | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` | Lịch sử perp đúng thời điểm và cube ngày |
| 1 | `qforecast/swing/features.py`, `signals.py` | Thống kê trượt, beta, xếp hạng chéo, thư viện tín hiệu |
| 2 | `qforecast/model/`, `trainer/` | Mô hình CNN-BiLSTM-attention, xuất ONNX |
| 2 | `qforecast/swing/book.py`, `strategy.py`, `paperbook.py`, `backtest.py` | Mô hình book, spec chiến thuật, book con so le, backtest kiểu sàn |
| 3 | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Kiểm định không rò rỉ, vault test sạch |
| 3 | `artifacts/models/c5.json` | Spec của C5 |
| 4 | `qforecast/adaptive/search.py`, `weights.py` | Tìm trong không gian trọng số, trọng số thích nghi mỗi ngày |
| 4 | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` | Tài khoản ảo, bot paper, dashboard trên máy |
| 5 | `qforecast/adaptive/engine.py` | Engine torch chạy nhiều bộ trọng số cùng lúc |

## Chạy nhanh

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # dựng cube ngày
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # backtest kiểu sàn của C5
python -m qforecast.swing.paper --plan                              # lệnh C5 hôm nay (không giao dịch)
python -m qforecast.swing.dashboard                                 # sổ paper + lịch chạy mỗi ngày
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # trọng số thích nghi cho một đoạn ngày
pytest -q
```

## Nhật ký xây dựng

| Chương | Ngày | Nhánh |
|---|---|---|
| 1. Pipeline dữ liệu | 2026-11-10 → 11-12 | `step-01/data-pipeline`, `step-02/signal-features` |
| 2. Mô hình trên dữ liệu | 2026-11-13 → 11-15 | `step-03/book-model` |
| 3. Thử nghiệm mô hình | 2026-11-16 → 11-19 | `step-04/research-toolkit`, `step-05/strategy-c5` |
| 4. Thuật toán thích nghi | 2026-11-20 → 11-22 | `step-06/adaptive-weights`, `step-07/paper-dashboard` |
| 5. Tốc độ | 2026-11-23 → 11-25 | `step-08/fast-adaptive-engine` |

## Luật code tuân theo

| Luật | Kiểm ở đâu |
|---|---|
| Quyết định lúc đóng cửa ngày k, giao dịch lúc mở cửa ngày k + 1 | `qforecast/swing/cube.py` (một quy ước thời gian cho mọi module) |
| Không thiên lệch sống sót: coin delist vẫn có trong xếp hạng cũ | `select_universe` chỉ dùng các dòng trước ngày đầu tháng |
| Không nhìn trước | Test cắt cube tại dòng k. Tín hiệu, điểm, mục tiêu và trọng số thích nghi của dòng k phải giữ nguyên |
| Không ăn may ngày đổi lệnh | Kết quả là trung bình các book con theo thứ trong tuần |
| Luôn tính phí | Phí mỗi chiều + funding thật. Bot paper khớp lệnh qua sổ lệnh thật |
| Lịch sử chỉ để loại | Biến thể phải hơn bản gốc ở 5 / 6 khối thời gian có purge. Chỉ dữ liệu chưa thấy (paper, vault niêm phong) mới xác nhận được |

Trọng số và số coin mỗi chân của C5 được chọn trên dữ liệu 2021-2026. Vì vậy backtest trên đoạn đó là trong mẫu:
nó cho thấy code chạy đúng, không phải dự báo lợi nhuận.

## Test

`pytest -q` chạy offline trên dữ liệu giả. Test kiểm:

- chặn lỗi tải dữ liệu, universe đúng thời điểm;
- không nhìn trước ở tín hiệu, mục tiêu và trọng số thích nghi;
- toán của book, luật chân của C5;
- engine chạy theo lô so với book tham chiếu;
- một bước paper có chạy bù, qua API của dashboard.
