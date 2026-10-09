[English](RESULTS_BENCHMARK.md) | **Tiếng Việt**

# Kết quả benchmark

Mọi con số đến từ một máy, và nếu không ghi chú khác thì từ một phiên đo ngày
**2026-10-09 (16:00–16:20)**:
laptop Intel Core i7-13700H (6 P-core có Hyper-Threading = CPU logic 0–11, 8 E-core = CPU
logic 12–19), Windows 11, Python 3.13.

Báo cáo gốc do script sinh tự động (chỉ có tiếng Anh):
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md): micro-benchmark từng thành phần;
- [benchmarks/BEFORE_AFTER.md](benchmarks/BEFORE_AFTER.md): pipeline prototype so với bản tối
  ưu, đo xen kẽ;
- [benchmarks/HETERO.md](benchmarks/HETERO.md): điều phối tác vụ hàng loạt trên P-core và
  E-core.

---

## 0. Cách đọc các con số

- **p50** là trung vị; **p99** là độ trễ mà 99% số lần gọi nhanh hơn. **Tăng tốc** là
  `trước ÷ sau` của p50.
- **Số tuyệt đối trên laptop dao động 2–3 lần giữa các phiên** (nhiệt độ, quản lý năng lượng,
  cách hệ điều hành xếp lịch). Ví dụ, việc tính lại features bằng pandas đo được 5.5 ms,
  6.9 ms và 18.5 ms ở ba phiên khác nhau. **Tỷ lệ thì ổn định**, nên mọi so sánh dưới đây
  đều là tỷ lệ đo trong cùng một phiên.
- Phép so sánh end-to-end (§1) chạy cả hai phiên bản **xen kẽ từng nến trong cùng một
  process**, nên cả hai cùng chịu một trạng thái máy. Hệ số tăng tốc kèm **khoảng tin cậy 95%**
  tính bằng bootstrap.
- ✅ = đã đo · 📐 = tính từ kích thước, không bấm giờ.

---

## 1. End-to-end: pipeline prototype so với bản tối ưu ✅

**Tác vụ:** đường nóng xử lý mỗi nến (giải mã frame → features → chuẩn hóa → inference →
đóng gói kết quả), 300 nến ETHUSDT thật. Mỗi nến đi qua cả hai phiên bản. Nguồn:
[BEFORE_AFTER.md](benchmarks/BEFORE_AFTER.md).

| Chính sách CPU | Trước (cách làm của prototype) | Sau (đã tối ưu) | Tăng tốc (KTC 95%) |
|---|---:|---:|---:|
| Mọi CPU | 56.6 ms | 1.07 ms | **52.7×** (49.0–56.0) |
| Chỉ P-core | 35.2 ms | 0.60 ms | **58.3×** (55.5–60.8) |

Từng giai đoạn (mọi CPU):

| Giai đoạn | Trước | Sau | Tăng tốc |
|---|---:|---:|---:|
| Giải mã frame | 32.0 µs | 22.4 µs | 1.4× |
| Features | 19,744 µs | 40.9 µs | **483×** |
| Chuẩn hóa + tạo input | 152 µs | 63.6 µs | 2.4× |
| Inference mô hình | 35,608 µs | 902 µs | **39.5×** |
| Đóng gói kết quả | 106 µs | 33.5 µs | 3.2× |

**Thời gian được dùng vào đâu:**
- **Trước:** features 35–44%, inference PyTorch 56–63%, mọi thứ còn lại dưới 1%.
- **Sau:** inference 81–84%, mọi thứ còn lại 16–19%.

Các phiên trước đo cùng phép so sánh này được 57× (55–70) và 46× (45–54): cùng bậc độ lớn,
khác tốc độ tuyệt đối.

Mỗi update sổ lệnh (giải mã + đóng gói lên bus): 4.7 µs → 1.1 µs (**4.3×**). Thông lượng
một luồng tăng từ 213k lên 909k message/giây, và message nhỏ đi từ 132 byte (JSON) xuống
60 byte (msgpack).

---

## 2. Train và inference PyTorch chưa tối ưu mất thời gian ở đâu ✅

### 2.1 Inference: cùng một mô hình qua các runtime khác nhau

Batch 1, input (1, 128, 14), 158k tham số. Nguồn: [RESULTS.md](benchmarks/RESULTS.md).

| Runtime | p50 | So với PyTorch mặc định |
|---|---:|---:|
| PyTorch eager, 12 thread (kiểu mặc định) | 4,224 µs | 1.0× |
| PyTorch eager, 1 thread | 665 µs | 6.4× |
| TorchScript (đã freeze), 1 thread | 535 µs | 7.9× |
| ONNX Runtime, 12 thread | 293 µs | 14.4× |
| **ONNX Runtime, 1 thread** | **181 µs** | **23.4×** |

Hai kết luận:
- Cùng phép tính chạy được trong 181 µs nghĩa là **hơn 95% của lần gọi eager 4.2 ms là chi
  phí phụ**: điều phối của Python cho từng phép toán, và việc đánh thức, đồng bộ thread pool
  cho các phép nhân ma trận rất nhỏ.
- **Một thread thắng mười hai thread** khi inference batch 1, ở cả PyTorch lẫn ONNX Runtime.

Kiến trúc cũng ảnh hưởng. BiLSTM 2 lớp trên 64 bước mất 472 µs trên ONNX Runtime 1 thread;
BiLSTM 1 lớp trên 32 bước, đặt sau hai lớp conv stride 2, chỉ mất 181 µs (**2.6×**), với chất
lượng dự báo tương đương.

### 2.2 Train: số thread

Một bước train mô hình LSTM (batch 512, forward + backward + AdamW), mỗi cấu hình đo 3 lần.
Chạy lại bằng `python -m benchmarks.bench_train_threads`.

| Số thread | ms mỗi bước, trung vị (min–max) | Mẫu/giây | So với mặc định của torch |
|---:|---:|---:|---:|
| 1 | 274 (267–313) | 1,867 | 0.61× |
| 2 | 267 (162–335) | 1,920 | 0.63× |
| 4 | 151 (115–221) | 3,396 | 1.12× |
| 6 | 146 (132–238) | 3,499 | 1.15× |
| 8 | 206 (199–216) | 2,489 | 0.82× |
| 12 | 171 (171–173) | 2,991 | 0.98× |
| **14 (mặc định của torch)** | **168** (157–169) | 3,041 | 1.00× |
| 20 | 158 (157–161) | 3,246 | 1.07× |

Điều này cho thấy:
- **14 thread chỉ nhanh hơn 1 thread 1.6 lần**, hiệu suất song song khoảng 12%. Phần lớn sức
  tính của các core thêm vào bị tiêu cho việc đồng bộ và cho phần hồi quy tuần tự của LSTM.
- **Nhiều thread hơn không phải lúc nào cũng nhanh hơn**: 8 thread chậm hơn 4 hoặc 6 thread.

Profile của cùng bước train (`python -m benchmarks.profile_train`) cho thấy khoảng 55% thời
gian nằm ở các kernel LSTM forward và backward của oneDNN, và khoảng 1% ở việc tải dữ liệu.

### 2.3 Train: kiến trúc và pipeline dữ liệu

| | Trước | Sau | Thay đổi |
|---|---:|---:|---:|
| Thời gian mỗi mẫu train (BiLSTM 2 lớp/64 bước → 1 lớp/32 bước), lấy từ log train | ~2.3 ms | ~0.24 ms | ~9× |
| Bộ nhớ cho 90k cửa sổ train (mảng N×128×14 tạo sẵn → cắt batch theo chỉ số) 📐 | ~645 MB | ~7 MB | ~90× |

---

## 3. Micro-benchmark từng thành phần ✅

Nguồn: [RESULTS.md](benchmarks/RESULTS.md).

### 3.1 Features cho mỗi nến mới

| Cách làm | p50 | Tăng tốc |
|---|---:|---:|
| pandas tính lại trên 500 nến (prototype) | 5,519 µs | 1× |
| numba, vẫn tính lại 500 nến | 113 µs | 49× (nhờ **biên dịch**) |
| **numba incremental, chỉ nến mới** | **3.1 µs** | **1,780×** (thêm nhờ **thuật toán**) |

Kernel incremental dùng chung cho train và chạy thật. Test kiểm tra nó khớp từng bit với đường
offline và với một bản pandas viết độc lập.

### 3.2 Định dạng truyền tin: mã hóa + giải mã một message

| Message | json | orjson | **msgspec msgpack** | Kích thước, json → msgpack |
|---|---:|---:|---:|---:|
| Kline | 6.3 µs | 1.1 µs | **0.6 µs** (10.5×) | 263 → 104 B |
| BookTicker | 4.5 µs | 0.7 µs | **0.4 µs** (11.2×) | 137 → 64 B |

### 3.3 Giải mã frame WebSocket

Từ `orjson → dict → float(str) → struct` sang giải mã có kiểu bằng `msgspec`:

| Frame | Trước | Sau | Tăng tốc |
|---|---:|---:|---:|
| bookTicker | 1.4 µs | 0.8 µs | 1.75× |
| aggTrade | 1.3 µs | 1.0 µs | 1.3× |
| kline | 2.1 µs | 1.7 µs | 1.24× |

### 3.4 Lưu trữ lịch sử nến

130,228 nến × 11 cột:

| Định dạng | Kích thước | Ghi | Đọc | Không mất dữ liệu |
|---|---:|---:|---:|:---:|
| `.npy` float64 (trước) | 11.46 MB | 5 ms | 5.1 ms | ✓ |
| `.npy` memory-map | 11.46 MB | 5 ms | 0.45 ms (đọc lười) | ✓ |
| Parquet + zstd | 8.09 MB | 105 ms | 14.5 ms | ✓ |
| QCOL, 1 thread | 2.88 MB | 347 ms | 17.5 ms | ✓ |
| **QCOL, giải mã song song theo cột** | **2.88 MB** (nhỏ hơn 4.0×) | 347 ms | 10.6 ms | ✓ |

QCOL lưu mỗi cột theo chuỗi `float64 → int64 có hệ số chính xác → delta (nếu có lợi) →
byte-shuffle → zstd`. Đọc chậm hơn `.npy` khoảng 2 lần; đó là cái giá có chủ đích để file nhỏ
hơn 4 lần.

### 3.5 Ghi tick (service `recorder`)

Luồng Binance thật đã ghi lại, 7,273 message:

| Chỉ số | Giá trị |
|---|---:|
| Thêm một message trên event loop, p50 / p99 | 0.4 / 0.5 µs |
| Nén + ghi trên thread I/O | 132 MB/s ≈ 1.8 triệu msg/s |
| Đọc lại + giải nén | 2.6 triệu msg/s |
| Trên đĩa mỗi message | **16.2 B** (nhỏ hơn msgpack 4.4×, nhỏ hơn JSON khoảng 8×) |
| Trên đĩa mỗi update sổ lệnh | **12.7 B** so với 146 B nếu lưu JSON thô (11.5×) |

---

## 4. Xếp lịch CPU trên chip hybrid ✅

### 4.1 P-core so với E-core, từng nhân một

| Tác vụ | P-core | E-core | E-core chậm hơn |
|---|---:|---:|---:|
| `json.dumps`, Python đơn luồng | 1.43–1.69 µs | 4.0–4.5 µs | ~2.7–3× |
| Inference ONNX, vòng lặp nóng | 204 µs | 463 µs | 2.3× |
| Inference ONNX ngay sau khi ngủ 1 ms | 426 µs | 2,013 µs | 4.7× |

(Số liệu này đo ở một phiên trước, ngày 2026-10-08/09.)

Khi chạy đầy tải, một CPU *logic* của P-core phải chia nhân vật lý với luồng Hyper-Threading
anh em của nó:
- tính theo CPU logic, P-core chỉ bằng **1.00–1.48×** E-core (0.86–1.83× qua các phiên);
- tính theo nhân vật lý, P-core bằng **2.0–3.0×** E-core.

### 4.2 Tác vụ hàng loạt: ghim + hàng đợi động so với Windows xếp lịch

**Tác vụ:** 20 tiến trình worker, mỗi CPU logic một tiến trình. Mỗi lần chạy xử lý một khối
việc cố định:
- **inference:** 132k cửa sổ mô hình;
- **ticks:** 13.8 triệu frame Binance.

Chỉ số đo là **makespan**: thời gian từ lúc thả mọi worker tới khi worker cuối cùng xong. Bảng
đầy đủ 11 chiến lược nằm trong [HETERO.md](benchmarks/HETERO.md). Các dòng quan trọng:

| Chiến lược | inference | ticks | Tỷ lệ rảnh |
|---|---:|---:|---:|
| Mọi core, chia đều, Windows xếp lịch (mốc) | 2.81 s | 2.88 s | 9–12% |
| Mọi core, **hàng đợi động**, Windows xếp lịch | 2.58 s (1.09×) | 2.68 s (1.08×) | 1% |
| Mọi core, hàng đợi động, Windows + tắt EcoQoS | 2.94 s (0.96×) | 2.52 s (1.14×) | 1% |
| **Ghim**, chia đều | 3.30 s (**0.85×**) | 3.26 s (**0.88×**) | 25–27% |
| Ghim, chia theo tốc độ | 2.57 s (1.09×) | 2.94 s (0.98×) | 9–15% |
| **Ghim + hàng đợi động** | 2.61 s (1.08×) | 2.64 s (1.09×) | 1–2% |
| Ghim + hàng đợi động + tắt EcoQoS | 2.44 s (1.15×) | 2.73 s (1.05×) | 1% |
| Chỉ P-core, hàng đợi động | 3.40 s (0.83×) | 4.06 s (0.71×) | 1% |

**Ghim + hàng đợi động so với Windows xếp lịch + hàng đợi động, qua các phiên:**

| Phiên | Windows đã làm gì | inference | ticks |
|---|---|---:|---:|
| A (phiên bản báo cáo trước) | **EcoQoS dồn worker lên E-core** (P-core bận 12–13%) | 8.54 s → 2.61 s (**3.27×**) | 5.98 s → 2.30 s (**2.60×**) |
| B | dùng đủ mọi core | 2.84 s → 2.83 s (1.00×) | 3.94 s → 3.57 s (1.10×) |
| C (phiên này, [HETERO.md](benchmarks/HETERO.md)) | dùng đủ mọi core | 2.58 s → 2.61 s (0.99×) | 2.68 s → 2.64 s (1.02×) |

**Hàng đợi động so với chia đều, cả hai đều ghim**, qua các phiên A–C: nhanh hơn
**1.07–1.46×**.

**Kết luận:**
1. **Hàng đợi động là cách xếp việc nên dùng.** Không cần hiệu chỉnh, giữ thời gian rảnh ở mức
   1–2%, và tốt nhất hoặc ngang tốt nhất ở mọi phiên.
2. **Ghim + hàng đợi động so với Windows + hàng đợi động:**
   - khi Windows dùng đủ mọi core, hai cách **tương đương** (0.92–1.21× qua các phiên và loại
     tác vụ, phần lớn nằm trong nhiễu);
   - khi EcoQoS dồn worker chạy nền lên E-core, ghim thắng **2.6–3.3 lần**, vì affinity mask
     ép việc chạy trên P-core.

   Vì vậy ghim là lựa chọn mặc định an toàn hơn.
3. **Ghim mà vẫn chia đều là lựa chọn tệ nhất.** Worker trên P-core xong sớm rồi rảnh 25–27%
   thời gian để chờ worker trên E-core; còn tệ hơn để Windows tự di chuyển việc.
4. **Chia theo tốc độ đo trước không đáng tin**: 0.99–1.28× so với ghim + chia đều qua các
   phiên, vì tốc độ hiệu chỉnh trước khi chạy bị lệch trong lúc chạy (Hyper-Threading, nhiệt
   độ).
5. **Không tắt E-core cho tác vụ hàng loạt.** Chỉ dùng P-core đạt 0.71–0.83× so với mốc.

Mỗi chiến lược có 3–6 lần đo mỗi phiên, và khoảng min–max của các dòng gần nhau thường chồng
lên nhau. Chênh lệch dưới khoảng 10% nên xem là xu hướng.

### 4.3 Phát hiện EcoQoS ✅

**EcoQoS** (power throttling) của Windows 11 có thể gắn nhãn tiết kiệm năng lượng cho tiến
trình chạy nền, và bộ xếp lịch hybrid khi đó giữ chúng trên E-core trong khi P-core ngồi
không. Phiên chẩn đoán (2026-10-09): inference, 20 tiến trình không ghim, mỗi nhóm tiến trình
mới chạy 4 lần liên tiếp.

| Cấu hình | Makespan của 4 lần chạy | P-core bận |
|---|---|---:|
| Mặc định, nhóm 1 | 3.40 · 6.56 · 6.46 · 6.09 s | 11–27% |
| Mặc định, nhóm 2 | 7.30 · 7.63 · 7.87 · 9.00 s | 15–20% |
| Tắt EcoQoS, nhóm 1 | 2.90 · 2.84 · 2.81 · 2.94 s | 97–100% |
| Tắt EcoQoS, nhóm 2 | 3.31 · 3.17 · 2.96 · 3.02 s | 97–100% |

Hiện tượng **phụ thuộc trạng thái máy**: nó không xảy ra ở phiên B và C. Các giải thích đã bị
bác bỏ, ghi lại để tham khảo:
- "tiến trình đã ghim rồi bỏ ghim thì bị chậm";
- core parking (giữ core thức bằng busy-wait không giúp gì);
- mức ưu tiên tiến trình (ABOVE_NORMAL và HIGH không giúp gì).

Các service giờ tự tắt EcoQoS khi khởi động (`qforecast/core/cpu.py`, mặc định
`QF_HIGH_QOS=1`).

### 4.4 Pipeline realtime: chính sách CPU

Replay 181,500 message qua đúng hai service `signal` và `analytics` thật, trên một event loop:

| Phiên | Mọi CPU | Chỉ P-core | Nến → forecast p50, mọi CPU → P-core |
|---|---:|---:|---:|
| 2026-10-08 | 18.4k msg/s | 60.0k msg/s (**3.3×**) | 1,901 → 655 µs |
| 2026-10-09 sáng | 77.4k msg/s | 102.9k msg/s (1.3×) | 426 → 393 µs |
| 2026-10-09 (phiên này) | 76.2k msg/s | 77.0k msg/s (1.0×) | 508 → 524 µs |

Giới hạn service realtime vào P-core giúp **loại bỏ rủi ro dao động giữa các lần chạy**, chứ
không thêm tốc độ cố định: khi hệ điều hành đặt event loop lên E-core thì thiệt hại lên tới
3 lần, còn khi không thì không có khác biệt.

---

## 5. Những gì chưa đo

| Hạng mục | Lý do |
|---|---|
| Thời gian đi qua Redis giữa các service (chế độ phân tán) | Docker chưa chạy trên máy phát triển; Redis bus mới chỉ được test bằng `fakeredis` |
| Độ trễ mạng từ sàn | lệch đồng hồ giữa sàn và máy khiến độ trễ một chiều không đáng tin |
| Tinh chỉnh trên Linux (`isolcpus`, `SCHED_FIFO`, governor hiệu năng) | máy phát triển chỉ chạy Windows |
| Inference live khi core "nguội" (1 nến mỗi phút) | quan sát được khoảng 0.7–0.9 ms p50 so với 0.2–0.3 ms trong vòng lặp nóng; chưa đo có hệ thống |
