[English](README.md) | **Tiếng Việt**

# Bộ giả lập giao dịch crypto và công cụ nghiên cứu

Dự án là bộ giả lập giao dịch theo ngày, xếp hạng chéo nhiều coin, cho perp USDT-M trên Binance. Gồm:

- công cụ nghiên cứu để không tự lừa mình;
- **C5**, chiến thuật swing long/short;
- **thuật tối ưu trọng số tín hiệu thích nghi**;
- bot giao dịch giấy (paper) kèm dashboard nhỏ chạy trên máy.

Dự án bắt đầu từ pipeline độ trễ thấp cho nến phút, giữ lại ở
[docs/LOW_LATENCY_PIPELINE.vi.md](docs/LOW_LATENCY_PIPELINE.vi.md). Model nến phút không có lợi thế sau phí. Vì vậy
dự án chuyển sang book theo ngày, và kiểm tra ý tưởng cẩn thận trước khi tin.

> Chỉ giao dịch giấy. Không có lệnh thật nào được gửi đi. Đây không phải lời khuyên đầu tư.

## Bên trong có gì

| Phần | Module | Làm gì |
|---|---|---|
| Dữ liệu | `qforecast/exchange/vision.py`, `qforecast/swing/data.py` | Tải mọi perp USDT từng niêm yết (cả coin đã delist) từ data.binance.vision và REST. Dựng cube ngày top-40 theo khối lượng, xếp lại mỗi tháng |
| Tín hiệu | `qforecast/swing/signals.py`, `features.py` | 17 tín hiệu ngày từ các bài nghiên cứu (momentum, xu hướng, funding đông, biến động thấp, MAX, quy mô, beta, dòng lệnh taker). Mỗi ngày xếp hạng giữa các coin |
| Bộ giả lập | `qforecast/swing/book.py`, `strategy.py` | Lô long/short, chân theo nghịch đảo biến động, hedge beta, chặn short bị squeeze, phí và funding. 7 book con so le theo tuần |
| Backtest kiểu sàn | `qforecast/swing/backtest.py`, `paperbook.py` | Cùng mục tiêu, chạy lại trên tài khoản USDT: khớp giá mở hôm sau, lệnh tối thiểu, lãi lỗ từng coin |
| Công cụ nghiên cứu | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Walk-forward có purge, CPCV, Sharpe giảm phát (DSR), PBO, bootstrap theo cặp, sàng 6 khối thời gian, vault niêm phong cho test sạch |
| Chiến thuật C5 | `artifacts/models/c5.json`, [docs/C5.md](docs/C5.md) | Điểm 6 tín hiệu, 5 long / 8 short, bỏ short có funding âm, hedge beta |
| Trọng số thích nghi | `qforecast/adaptive/`, [docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md) | Engine torch chạy khoảng 120 000 bộ trọng số cùng lúc. Mỗi ngày chọn bộ tốt nhất 30 ngày qua, rồi tinh chỉnh quanh nó |
| Bot paper + dashboard | `qforecast/swing/paper.py`, `account.py`, `dashboard.py` | Tài khoản ảo khớp trên sổ lệnh thật. Máy tắt thì chạy bù. Trang xem tại 127.0.0.1:8765 |

## Chạy nhanh

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # cube ngày, lần đầu ~10-20 phút
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # backtest kiểu sàn của C5
python -m qforecast.swing.paper --plan                              # lệnh C5 hôm nay (không giao dịch)
python -m qforecast.swing.dashboard                                 # sổ paper + lịch chạy mỗi ngày
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # trọng số thích nghi (CPU: vài phút mỗi lần)
pytest -q
```

## Xây dựng thế nào

Mỗi bước là một nhánh, có pull request nối lên nhánh trước.

| Ngày | Bước | Nhánh |
|---|---|---|
| 2026-11-10 → 11-12 | Công cụ nghiên cứu trước: purged CV, CPCV, PSR / DSR, PBO, bootstrap, sàng khối, vault | `step-01/validation-toolkit` |
| 2026-11-13 | Dữ liệu đúng thời điểm: mọi perp từng niêm yết, top-40 mỗi tháng, cube ngày | `step-02/daily-data-cube` |
| 2026-11-14 | Thư viện tín hiệu kèm chiều mỗi bài báo dự đoán, test không nhìn trước | `step-03/signal-library` |
| 2026-11-15 → 11-17 | Bộ giả lập book, spec chiến thuật, book con so le, backtest kiểu sàn | `step-04/book-simulator` |
| 2026-11-18 → 11-19 | C5: hai chân lệch số coin, bỏ short funding âm, giữ trần khi chân mỏng | `step-05/strategy-c5` |
| 2026-11-20 → 11-21 | Trọng số thích nghi: engine chạy theo lô, bộ phủ + tinh chỉnh, cache ngày | `step-06/adaptive-weights` |
| 2026-11-22 → 11-25 | Bot paper trên sổ lệnh thật, dashboard, tài liệu | `step-07/paper-dashboard` |

## Luật code tuân theo

| Luật | Kiểm ở đâu |
|---|---|
| Quyết định lúc đóng cửa ngày k, giao dịch lúc mở cửa ngày k + 1 | `qforecast/swing/cube.py` (một quy ước thời gian cho mọi module) |
| Không thiên lệch sống sót: coin delist vẫn có trong xếp hạng cũ | `select_universe` chỉ dùng các dòng trước ngày đầu tháng |
| Không nhìn trước | Test cắt cube tại dòng k. Tín hiệu, điểm, mục tiêu và trọng số thích nghi của dòng k phải giữ nguyên |
| Không ăn may ngày đổi lệnh | Kết quả là trung bình 7 book con theo thứ trong tuần |
| Luôn tính phí | 7 bps mỗi chiều + funding thật. Bot paper khớp lệnh qua sổ lệnh thật |
| Lịch sử chỉ để loại | Biến thể phải hơn bản gốc ở 5 / 6 khối thời gian có purge. Chỉ dữ liệu chưa thấy (paper, vault niêm phong) mới xác nhận được |

Trọng số và số coin mỗi chân của C5 được chọn trên dữ liệu 2021-2026. Vì vậy backtest trên đoạn đó là trong mẫu:
nó cho thấy code chạy đúng, không phải dự báo lợi nhuận.

## Test

`pytest -q` chạy offline trên dữ liệu giả. Test kiểm:

- chặn lỗi tải dữ liệu, universe đúng thời điểm;
- không nhìn trước ở tín hiệu và mục tiêu;
- toán của book, luật chân của C5;
- engine torch so với book numpy (sai số 1e-12);
- trọng số thích nghi không rò rỉ và tái lập được;
- một bước paper có chạy bù, qua API của dashboard.

## Phần trước đó

Pipeline độ trễ thấp ban đầu vẫn còn trong repo: feature tăng dần, CNN-BiLSTM-attention trên ONNX Runtime,
microservice qua Redis Streams, benchmark CPU / I/O. Xem
[docs/LOW_LATENCY_PIPELINE.vi.md](docs/LOW_LATENCY_PIPELINE.vi.md) và [RESULTS_BENCHMARK.vi.md](RESULTS_BENCHMARK.vi.md).
