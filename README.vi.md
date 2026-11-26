[English](README.md) | **Tiếng Việt**

# Giao dịch crypto thích nghi: từ dữ liệu thô đến thuật toán thích nghi tốc độ cao

[![ci](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml/badge.svg)](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml)

Dự án nghiên cứu cá nhân trên perp USDT-M của Binance. Gồm pipeline dữ liệu realtime, chiến thuật long/short xếp
hạng chéo (**C5**), bộ công cụ kiểm định để không tự lừa mình, và **thuật toán trọng số tín hiệu thích nghi** chạy
trên engine torch theo lô. Tất cả chạy ở chế độ giao dịch giấy.

## Điểm nổi bật

- **Dữ liệu đúng thời điểm.** Giữ coin đã delist, lọc coin chết, universe chỉ xếp hạng bằng dữ liệu quá khứ. Có
  test chứng minh không tín hiệu nào nhìn thấy tương lai.
- **Giả lập sát sàn thật.** Có phí, funding, lệnh tối thiểu, hedge beta, chặn short bị squeeze, và book con so le
  để kết quả không phụ thuộc ngày bắt đầu.
- **Kiểm định trung thực.** CV có purge và CPCV, Sharpe giảm phát, PBO, sàng lọc nhiều khối thời gian, vault niêm
  phong cho test sạch.
- **Thuật toán thích nghi.** Trọng số tín hiệu được tối ưu lại mỗi ngày trên cửa sổ trượt, với luật thời gian chặt,
  không nhìn trước.
- **Tối ưu hiệu năng.** Việc tìm trọng số chạy trên engine torch theo lô (CPU / GPU). Bản tham chiếu chậm hơn vẫn
  được giữ, và test kiểm hai bản cho cùng lãi lỗ.

## Dự án đã phát triển thế nào

Mỗi bước trả lời một nhu cầu mà bước trước còn để ngỏ.

| Bước | Câu hỏi | Đã xây |
|---|---|---|
| 1 | Làm sao làm sạch dữ liệu thị trường và tính toán nhanh? | Pipeline realtime, cube ngày đúng thời điểm, thư viện tín hiệu |
| 2 | Mô hình trên dữ liệu này có ra quyết định được không? | Mô hình CNN-BiLSTM nến phút, rồi mô hình book long/short với backtest kiểu sàn |
| 3 | Làm sao phân biệt lợi thế thật với may mắn? | Bộ công cụ kiểm định, rồi chiến thuật C5 xây theo luật của bộ đó |
| 4 | Chiến thuật có thích nghi theo thị trường được không? | Trọng số tín hiệu thích nghi mỗi ngày, chạy trên bot paper có dashboard |
| 5 | Việc tìm thích nghi có nhanh đủ để tìm kỹ không? | Engine torch theo lô, tổng cộng dồn thay cho mô phỏng lại, hỗ trợ GPU |

Mỗi bước là một nhánh, từ `step-01/data-pipeline` đến `step-08/fast-adaptive-engine`, đã merge vào main theo thứ tự.

## Kiến trúc

| Tầng | Module |
|---|---|
| Pipeline realtime | `qforecast/services/`, `core/`, `storage/`, `model/` |
| Dữ liệu | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` |
| Tín hiệu | `qforecast/swing/features.py`, `signals.py` |
| Chiến thuật và giả lập | `qforecast/swing/book.py`, `strategy.py`, `backtest.py`, `artifacts/models/c5.json` |
| Kiểm định | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` |
| Thuật toán thích nghi | `qforecast/adaptive/search.py`, `weights.py`, `engine.py` |
| Giao dịch giấy | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` |

## Chạy nhanh

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data              # dựng cube ngày
python -m qforecast.swing.backtest          # backtest C5
python -m qforecast.swing.paper --plan      # lệnh hôm nay (không giao dịch)
python -m qforecast.swing.dashboard         # sổ paper tại http://127.0.0.1:8765
pytest -q                                   # test offline trên dữ liệu giả
```

## Hiện trạng

- C5 và book thích nghi đang chạy paper, đây là test sạch duy nhất còn lại.
- Trong test walk-forward, trọng số thích nghi chưa thắng trọng số cố định. Nó được giữ như một book nghiên cứu.
- Trọng số của C5 được chọn trên chính đoạn lịch sử mà backtest dùng. Vì vậy backtest cho thấy code chạy đúng, không
  cho biết nên kỳ vọng gì.

## Tài liệu

[Chiến thuật C5](docs/C5.md) · [Trọng số thích nghi](docs/ADAPTIVE_WEIGHTS.md) ·
[Pipeline độ trễ thấp](docs/LOW_LATENCY_PIPELINE.vi.md) · [Benchmark](RESULTS_BENCHMARK.vi.md)

> Chỉ giao dịch giấy. Không có lệnh thật. Không phải lời khuyên đầu tư.
