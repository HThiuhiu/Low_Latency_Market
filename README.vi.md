[English](README.md) | **Tiếng Việt**

# qforecast: bộ nghiên cứu giao dịch crypto được xây để tự chứng minh chiến thuật của mình sai

[![ci](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml/badge.svg)](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B%20%28CI%203.12%29-blue)
![mode](https://img.shields.io/badge/trading-paper%20only-lightgrey)

Hệ thống nghiên cứu và giao dịch làm một mình, trọn từ đầu đến cuối, cho thị trường crypto trên Binance. Gồm ba
phần tách rời:

- **Pipeline realtime** (giai đoạn đầu của dự án): các service asyncio trên Redis Streams, nhận, chấm điểm và phát
  dữ liệu spot trực tiếp.
- **Bộ công cụ kiểm định:** các test cố bác bỏ mọi backtest trước khi tin nó.
- **Chiến thuật ngày** (C5; spec đã khóa ở [`artifacts/models/c5.json`](artifacts/models/c5.json)): danh mục
  long/short trung lập thị trường trên perp USDT-M (hợp đồng tương lai crypto không có ngày đáo hạn). Mỗi ngày nó
  xếp hạng các perp thanh khoản nhất theo các yếu tố đã công bố, như giá ít biến động, ít nhạy với BTC và chi phí
  giữ thấp. Nó mua nhóm đầu, bán khống nhóm cuối, và chỉnh cỡ hai bên để biến động của BTC triệt tiêu nhau. Nó giao
  dịch giấy, tự chạy, và không dùng pipeline realtime.

Xây bằng Python, PyTorch, một kernel Triton tự viết, Docker và GitHub Actions. Mô hình deep learning nến phút không
có lợi thế sau phí nên đã bị loại. Một chiến thuật ngày được giữ lại và đang chờ dữ liệu nó chưa từng thấy.

**Hiện trạng:** danh mục giấy chạy trên giá Binance thật từ tháng 11/2026, không có lệnh thật. Thời gian đó còn quá
ngắn để phân biệt kỹ năng với may mắn.

## Xem ở đâu

| Nếu bạn tuyển | Mở trước |
|---|---|
| Nghiên cứu quant | [`validation.py`](qforecast/validation.py), [`vault.py`](qforecast/vault.py), [`blockcv.py`](qforecast/blockcv.py), [chiến thuật C5](docs/C5.md) |
| ML / GPU | [`scan_triton.py`](qforecast/model/scan_triton.py), [`adaptive/engine.py`](qforecast/adaptive/engine.py), [trọng số thích nghi](docs/ADAPTIVE_WEIGHTS.md) |
| Backend / dữ liệu | [`services/`](qforecast/services/), [`swing/paper.py`](qforecast/swing/paper.py), [`exchange/vision.py`](qforecast/exchange/vision.py), [bài viết về pipeline](docs/LOW_LATENCY_PIPELINE.vi.md) |
| Quy trình làm việc | Các nhánh `step-01` … `step-08`: chiến thuật ngày được xây từng bước, mỗi bước xem lại được (xem cuối trang) |

## Điểm nhấn kỹ thuật

Dữ liệu spot trực tiếp đi theo ingestor → signal → analytics → gateway. Mỗi tầng là một service asyncio trên Redis
Streams. Chiến thuật ngày chạy như một tác vụ hẹn giờ riêng.

| Cái gì | Giải quyết vấn đề gì | Code và test |
|---|---|---|
| **Service hướng sự kiện:** tự khởi động lại với thời gian chờ tăng dần (có test); tải bù bar bị lỡ khi kết nối lại và số liệu Prometheus qua gateway (có trong code, chưa có test) | Một tầng bị sập hay một socket bị rớt không được làm dừng pipeline hoặc mất bar | [`services/all.py`](qforecast/services/all.py), [`ingestor.py`](qforecast/services/ingestor.py); [`test_supervisor.py`](tests/test_supervisor.py), [`test_bus.py`](tests/test_bus.py) |
| **Bỏ chi phí thừa:** feature tính tăng dần, ring buffer không sao chép, ONNX Runtime, gán luồng vào nhân | Chi phí thừa, không phải phép tính, chiếm phần lớn thời gian của pipeline ngây thơ | [bài viết](docs/LOW_LATENCY_PIPELINE.vi.md); [`test_features.py`](tests/test_features.py) (online bằng offline, không nhìn trước), [`test_ringbuffer.py`](tests/test_ringbuffer.py); CI chạy thử benchmark |
| **Kernel GPU tự viết:** kernel Triton gộp cho phép hồi quy của một lớp state-space (kiểu Mamba) tùy chọn, test trên GPU với bản tham chiếu qua gradcheck; giữ lại như một thành phần đã test sau khi mô hình bị loại vì không có lợi thế | Quét tuần tự làm phí GPU | [`scan_triton.py`](qforecast/model/scan_triton.py), [`test_scan.py`](tests/test_scan.py) |
| **Engine giả lập theo lô:** chạy chiến thuật cho nhiều bộ trọng số tín hiệu trong một lượt tensor, trên CPU hoặc GPU, giữ bản chậm làm bản đối chiếu | Tìm trọng số kỹ mỗi ngày mà chạy từng bộ một thì quá chậm | [`engine.py`](qforecast/adaptive/engine.py), [`test_adaptive_engine.py`](tests/test_adaptive_engine.py) |
| **Bot paper tự chạy:** lưu trạng thái, ngày nào máy tắt thì chạy bù ở giá mở cửa của ngày đó | Bot trên laptop phải sống qua lúc máy ngủ và khởi động lại mà không âm thầm bỏ lệnh | [`paper.py`](qforecast/swing/paper.py), [`test_swing_paper.py`](tests/test_swing_paper.py) |
| **Test có đáp án biết trước:** kiểm phần thống kê trên dữ liệu giả mà đáp án đúng đã biết (nhiễu thuần so với lợi thế cài sẵn, một file niêm phong bị sửa) | Code thống kê có thể sai mà trông vẫn đúng | [`test_validation.py`](tests/test_validation.py), [`test_evaluation.py`](tests/test_evaluation.py), [`test_vault.py`](tests/test_vault.py) |

## Đã thử gì, và kết luận

| Ý tưởng | Cách kiểm | Kết luận |
|---|---|---|
| Mô hình dự báo CNN-BiLSTM nến phút | Các giai đoạn giữ riêng, có tính phí giao dịch ([chi tiết](docs/LOW_LATENCY_PIPELINE.vi.md)) | Loại: không có lợi thế sau phí |
| Chiến thuật long/short ngày (C5) | Dữ liệu đúng thời điểm, bảy ngày bắt đầu so le, hiệu chỉnh theo số biến thể đã thử | Giữ làm ứng viên duy nhất; đang giao dịch giấy |
| [Trọng số thích nghi](docs/ADAPTIVE_WEIGHTS.md) (tối ưu lại mỗi ngày) | Walk-forward, đúng thời điểm | Không thắng trọng số cố định; giữ trên paper để so sánh |

### Các bẫy backtest và cách xử lý

| Bẫy backtest | Cách xử lý | Test |
|---|---|---|
| Thiên lệch sống sót: danh sách coin hôm nay chỉ còn coin sống sót | Tải mọi perp từng niêm yết; universe chỉ xếp hạng bằng dữ liệu trước mỗi tháng | [`test_swing_data.py`](tests/test_swing_data.py) |
| Nhìn trước: dùng thông tin chưa có lúc quyết định | Một luật thời gian chung: quyết định lúc đóng cửa, giao dịch lúc mở cửa hôm sau | [`test_swing_signals.py`](tests/test_swing_signals.py), [`test_c5.py`](tests/test_c5.py), [`test_adaptive.py`](tests/test_adaptive.py) |
| Bỏ qua phí và funding | Có phí, funding thật, và lệnh paper khớp qua sổ lệnh cho cả cỡ lệnh | [`test_swing_book.py`](tests/test_swing_book.py), [`test_swing_account.py`](tests/test_swing_account.py) |
| May rủi ngày bắt đầu | Trung bình của bảy danh mục con đổi lệnh vào các thứ khác nhau trong tuần | [`test_swing_book.py`](tests/test_swing_book.py) |
| Giữ lại biến thể may nhất trong nhiều lần thử | Sharpe giảm phát và xác suất backtest quá khớp (PBO) trừ hao kết quả theo số biến thể đã thử; bản thắng còn phải thắng ở phần lớn các khối thời gian | [`test_validation.py`](tests/test_validation.py), [`test_evaluation.py`](tests/test_evaluation.py) |
| Coi lịch sử là xác nhận | Lịch sử chỉ dùng để loại; xác nhận đến từ dữ liệu niêm phong hoặc dữ liệu tương lai | [`test_vault.py`](tests/test_vault.py) |

## Chạy thử

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt

# Bốn điều cốt lõi, chạy offline trên thị trường giả: không tín hiệu nào nhìn thấy tương lai, engine nhanh khớp bản
# tham chiếu, test quá khớp phân biệt được nhiễu với lợi thế, và dữ liệu niêm phong không tráo được.
pytest -q tests/test_swing_signals.py::test_every_signal_is_point_in_time \
          tests/test_adaptive_engine.py::test_engine_matches_the_reference_book \
          tests/test_validation.py::test_pbo_noise_vs_real_edge \
          tests/test_vault.py::test_vault_rejects_changed_cube

pytest -q                                   # toàn bộ test, offline
python -m qforecast.swing.data              # tải lịch sử và dựng dữ liệu ngày
python -m qforecast.swing.backtest          # backtest C5 trên tài khoản USDT: cỡ lệnh thật, phí, funding
python -m qforecast.swing.paper --plan      # lệnh hôm nay, không giao dịch
python -m qforecast.swing.dashboard         # danh mục giấy tại http://127.0.0.1:8765
```

Repo không chứa dữ liệu thị trường. Dữ liệu lấy từ endpoint và kho lưu trữ công khai của Binance, không cần API key.
Mỗi lần push, CI chạy `ruff`, `pytest`, chạy thử benchmark và build Docker. Test kernel GPU được bỏ qua trên máy
không có GPU NVIDIA.

## Quyết định chính và đánh đổi

- **Một giao diện bus, hai backend.** Khi triển khai dùng Redis Streams, để tầng nào khởi động lại thì đọc lại stream
  thay vì gọi lại sàn; bus trong RAM cùng giao diện, để test không cần server (Redis được giả lập bằng fakeredis).
- **Niêm phong dữ liệu sạch trước mọi kết luận.** Bộ dữ liệu niêm phong được kiểm hash, chỉ mở khi có file đăng ký
  trước, và chỉ mở một lần; mở lần hai thì bị đánh dấu đã dùng.
- **Giả lập khớp taker (lệnh thị trường).** Đây là lựa chọn thận trọng; khớp maker rẻ hơn nhưng chưa được chứng minh
  cho tới khi đo thật.
- **Giữ bản tham chiếu chậm sau khi viết engine nhanh.** Một test đối chiếu rẻ hơn một lỗi âm thầm trong việc tìm
  kiếm chạy mỗi ngày.

## Giới hạn và bước tiếp theo

| Giới hạn | Bước tiếp theo |
|---|---|
| Trọng số của C5 được chọn trên chính đoạn lịch sử mà backtest dùng, nên backtest chứng minh code chạy đúng, không cho biết nên kỳ vọng gì | Chấm cửa sổ tương lai niêm phong: dữ liệu ghi sau khi thiết kế đã khóa, khóa bằng hash để không tráo được |
| Chỉ giả lập khớp taker | Đo khớp maker (lệnh limit chờ, phí thấp hơn) bằng lệnh thật nhỏ trước khi dùng vốn thật |
| Trọng số thích nghi chưa được chứng minh | Giữ trên paper cạnh C5; chỉ đổi nếu dữ liệu tương lai ủng hộ |
| Chỉ dùng giá đóng cửa ngày và một sàn | Mô hình hóa squeeze trong ngày và thanh lý trước khi dùng vốn thật |

<details>
<summary><b>Dự án đã phát triển thế nào</b>: mỗi bước một nhánh, đọc được theo thứ tự</summary>

Mọi thứ đã merge vào nhánh mặc định. Pipeline và mô hình phút có trước. Sau đó chiến thuật ngày được xây trong tám
bước, mỗi bước một nhánh, bước sau lấp chỗ trống mà bước trước còn để lại.

| Nhánh | Thêm gì |
|---|---|
| `step-01/data-pipeline` | Lịch sử đủ mọi perp kể cả coin đã delist, universe mỗi tháng, dữ liệu ngày |
| `step-02/signal-features` | Tín hiệu từ các nghiên cứu đã công bố, xếp hạng giữa các coin mỗi ngày |
| `step-03/book-model` | Danh mục long/short với hedge thị trường, phí và funding; backtest trên tài khoản USDT |
| `step-04/research-toolkit` | Cross-validation có purge và tổ hợp, Sharpe giảm phát, PBO, sàng khối thời gian, vault niêm phong |
| `step-05/strategy-c5` | C5, xây theo các luật đó |
| `step-06/adaptive-weights` | Trọng số thích nghi, có test đúng thời điểm |
| `step-07/paper-dashboard` | Bot paper trên sổ lệnh thật, chạy bù ngày lỡ, dashboard trên máy |
| `step-08/fast-adaptive-engine` | Engine giả lập theo lô, có test đối chiếu với bản tham chiếu |

</details>

<details>
<summary><b>Cấu trúc repo</b></summary>

Repo được đặt tên theo giai đoạn đầu, pipeline độ trễ thấp. Package là `qforecast`.

```text
qforecast/   services/ core/ storage/ model/ trainer/ sim/ agent/   pipeline spot realtime, mô hình nến phút,
                                                                    và cố vấn LLM chỉ có quyền phủ quyết (chỉ paper)
             exchange/ swing/ adaptive/                             dữ liệu perp ngày, C5, bot paper, trọng số thích nghi
             validation.py evaluation.py blockcv.py vault.py        bộ công cụ kiểm định
tests/  benchmarks/  deploy/  docs/  research/  ModelTrade/ (prototype ban đầu)
```

Bài viết benchmark: [RESULTS_BENCHMARK.vi.md](RESULTS_BENCHMARK.vi.md).

</details>

> Chỉ nghiên cứu và giao dịch giấy. Không có lệnh thật nào được gửi đi. Không phải lời khuyên đầu tư.
