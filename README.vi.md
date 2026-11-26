[English](README.md) | **Tiếng Việt**

# Từ dữ liệu thị trường thô đến thuật toán giao dịch thích nghi

Dự án này không được lên kế hoạch từ đầu. Nó lớn lên từng câu hỏi một. Mỗi bước bắt đầu từ một nhu cầu mà bước
trước còn để ngỏ. Tôi tìm hiểu cái cần có để đáp ứng nó, xây cái đó, và câu trả lời lại mở ra câu hỏi tiếp theo.

> Chỉ giao dịch giấy. Không có lệnh thật nào được gửi đi. Đây không phải lời khuyên đầu tư.

## Nhìn nhanh cả hành trình

| Bước | Nhu cầu | Đã xây | Mở ra điều gì |
|---|---|---|---|
| 1 | Làm sạch dữ liệu sàn và tính toán nhanh trên nó | Pipeline realtime, rồi cube dữ liệu ngày đúng thời điểm và thư viện tín hiệu | Dữ liệu tin được, sẵn sàng cho mô hình |
| 2 | Biến dữ liệu thành quyết định | Mô hình dự báo nến phút, rồi mô hình book long/short với backtest kiểu sàn | Có mô hình, và câu hỏi: nó có tốt thật không |
| 3 | Phân biệt lợi thế thật với may mắn | Bộ công cụ kiểm định, rồi chiến thuật C5 xây theo luật của bộ đó | Một chiến thuật với trọng số tín hiệu cố định |
| 4 | Cho chiến thuật đi theo thị trường thay đổi | Trọng số tín hiệu tối ưu lại mỗi ngày, chạy trên bot paper có dashboard | Thuật toán chạy được nhưng tìm còn quá ít |
| 5 | Làm việc tìm thích nghi đủ nhanh để tìm kỹ | Engine chạy theo lô trên CPU / GPU, đánh giá nhiều bộ trọng số cùng lúc | Việc tìm phủ cả không gian trọng số mỗi ngày |

---

## Bước 1 · Làm sao làm sạch dữ liệu thị trường và tính toán nhanh?

**Ý tưởng.** Trước khi giao dịch, tôi muốn hiểu dữ liệu: lấy cái sàn gửi về, lọc nó, và tính toán trên nó mà không
phí thời gian.

**Đã thử.** Pipeline realtime cho dữ liệu Binance trực tiếp: feature tính tăng dần, ring buffer, microservice qua
Redis Streams. Tôi đo xem thời gian CPU và I/O đi đâu, rồi bỏ phần lãng phí. Chi tiết ở
[docs/LOW_LATENCY_PIPELINE.vi.md](docs/LOW_LATENCY_PIPELINE.vi.md).

**Học được.** Code nhanh mà dữ liệu sai thì vô ích. Có ba cái bẫy:

- coin đã delist biến mất khỏi danh sách hiện tại, nên lịch sử dựng từ danh sách đó chỉ còn coin sống sót;
- coin bị dừng giao dịch có thể giữ giá đứng yên, không có khối lượng, trong nhiều tuần;
- funding phải tính vào đúng ngày vị thế thật sự được giữ.

**Đã xây để đáp ứng.**

- `qforecast/exchange/vision.py`: mọi perp USDT từng niêm yết, cả coin đã delist, lấy từ kho lưu trữ công khai.
- `qforecast/swing/cube.py`, `data.py`: cube ngày với một luật thời gian chung (quyết định lúc đóng cửa, giao dịch
  lúc mở cửa hôm sau). Universe là top-40 mỗi tháng, chỉ xếp hạng bằng dữ liệu trước tháng đó. Coin chết bị lọc bỏ.
  Nếu thiếu một coin đang niêm yết thì dừng dựng, chứ không âm thầm bỏ qua.
- `qforecast/swing/features.py`, `signals.py`: 17 tín hiệu ngày từ các bài nghiên cứu, xếp hạng giữa các coin mỗi
  ngày. Test tính lại mọi tín hiệu trên cube cắt tại ngày đó, để chứng minh không có gì rò từ tương lai.

**Nhu cầu tiếp theo.** Dữ liệu sạch chỉ là phương tiện. Nó có dẫn tới quyết định được không?

## Bước 2 · Mô hình trên dữ liệu này có ra quyết định được không?

**Ý tưởng.** Xây mô hình trên dữ liệu và để nó chọn vị thế.

**Đã thử.** Đầu tiên là mạng CNN-BiLSTM-attention dự báo một coin trên nến phút, chạy bằng ONNX Runtime. Sau đó là
một kiểu mô hình khác: không dự báo một coin, mà mỗi ngày xếp hạng mọi coin với nhau, giữ nhóm tốt nhất và nhóm tệ
nhất.

**Học được.** Dự báo chưa phải là giao dịch. Phí, funding, cỡ vị thế và chiều thị trường quyết định kết quả không
kém gì tín hiệu. Muốn chấm một mô hình, tôi cần bộ giả lập giao dịch giống sàn thật.

**Đã xây để đáp ứng.**

- `qforecast/swing/book.py`: long nhóm đầu bảng, short nhóm cuối bảng. Cỡ mỗi chân theo nghịch đảo biến động, có
  trần mỗi coin. Giữ vị thế một tuần, tính phí và funding mỗi ngày, có chặn mua lại short khi coin tăng quá mạnh.
- `qforecast/swing/strategy.py`: một file spec JSON thành mục tiêu book cho mọi nơi dùng. Có hedge beta để book ăn
  chênh lệch giữa các coin, không ăn theo chiều thị trường. Book chia thành các book con theo thứ trong tuần, để kết
  quả không phụ thuộc ngày tình cờ bắt đầu.
- `qforecast/swing/backtest.py`, `paperbook.py`: cùng mục tiêu đó, chạy lại trên tài khoản USDT. Khớp ở giá mở hôm
  sau, có lệnh tối thiểu, có lãi lỗ từng coin.

**Nhu cầu tiếp theo.** Backtest đã ra con số. Làm sao biết đó không phải may mắn?

## Bước 3 · Làm sao phân biệt lợi thế thật với may mắn?

**Ý tưởng.** Thử nghiệm mọi mô hình một cách trung thực trước khi tin.

**Đã thử.** Cho mô hình nến phút chịu phí thật, và lợi thế biến mất. Book ngày cũng có thể đẹp vì lý do sai: nhãn
chồng lên nhau, thử quá nhiều lần, trúng một giai đoạn may.

**Học được.** Phần lớn ý tưởng thất bại, và backtest đẹp không phải là bằng chứng. Lịch sử chỉ dùng được để loại bỏ
ý tưởng. Chỉ dữ liệu mà lúc thiết kế chưa từng thấy mới xác nhận được.

**Đã xây để đáp ứng.**

- `qforecast/validation.py`: walk-forward có purge, CPCV, IC xếp hạng với t-stat trung thực, Sharpe giảm phát theo số
  lần thử, xác suất backtest quá khớp.
- `qforecast/evaluation.py`, `blockcv.py`: bootstrap theo cặp, và sàng lọc mà biến thể phải hơn bản gốc ở phần lớn
  các khối thời gian tách rời, không chỉ hơn ở trung bình.
- `qforecast/vault.py`: dữ liệu niêm phong, mỗi test chỉ được mở một lần, và chỉ sau khi đã ghi sẵn kế hoạch.
- **C5** (`artifacts/models/c5.json`, [docs/C5.md](docs/C5.md)): chiến thuật rút ra từ các lần thử nghiệm. Mỗi
  luật trả lời một rủi ro mà test đã chỉ ra:
  - chân short rộng hơn chân long, để một lệnh short bị bơm gây hại ít hơn;
  - bỏ short coin có funding âm, vì phía short ở đó đã đông;
  - book con so le và hedge beta bỏ đi may rủi về thời điểm và chiều thị trường.

**Nhu cầu tiếp theo.** C5 giữ trọng số tín hiệu cố định. Thị trường thì thay đổi. Trọng số có nên đổi theo không?

## Bước 4 · Chiến thuật có thích nghi theo thị trường được không?

**Ý tưởng.** Mỗi ngày chọn lại trọng số tín hiệu từ cái đã hiệu quả trong mấy tuần qua, rồi giao dịch ngày hôm sau
bằng nó.

**Đã thử.** Mỗi ngày tìm trong không gian trọng số. Một nhóm bộ trọng số rải khắp không gian được chấm điểm trên cửa
sổ trượt. Các bộ tốt nhất được tinh chỉnh quanh chỗ của nó, và bộ thắng giao dịch ngày hôm sau. Luật thời gian rất
chặt: trọng số của một ngày chỉ dùng lãi lỗ đã biết trước lúc mở cửa hôm sau. Test tính lại từng ngày trên cube kết
thúc đúng ngày đó.

**Học được.** Cách làm chạy được và trung thực, nhưng lộ ra hai giới hạn.

- Trong test walk-forward, trọng số thích nghi chưa thắng trọng số cố định. Nó chạy như một book nghiên cứu có theo
  dõi, không thay C5 ([docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md)).
- Bản đầu tiên chấm từng bộ trọng số lần lượt, qua cả book viết bằng Python. Kết quả đúng nhưng chậm, nên chỉ xem
  được một phần nhỏ của không gian trọng số.

**Đã xây để đáp ứng.**

- `qforecast/adaptive/search.py`, `weights.py`: việc tìm, trọng số mỗi ngày, và cache. Trọng số của một ngày không
  bao giờ đổi sau khi ngày đó đã giao dịch.
- `qforecast/swing/account.py`, `paper.py`, `dashboard.py`: bot paper chạy song song C5 và book thích nghi trên sổ
  lệnh thật, với tài khoản ảo, không cần API key. Ngày nào máy tắt thì chạy bù. Dashboard trên máy hiện cả hai book.

**Nhu cầu tiếp theo.** Việc tìm thích nghi chỉ tốt bằng phần không gian nó xem được. Nó phải nhanh.

## Bước 5 · Việc tìm thích nghi có nhanh đủ để tìm kỹ không?

**Ý tưởng.** Dùng lại bài học tốc độ của Bước 1: đo trước, rồi bỏ phần lãng phí. Lần này đối tượng là việc tìm
trọng số thích nghi.

**Đã thử.**

| Thay đổi | Vì sao |
|---|---|
| Bộ trọng số thành một chiều của tensor | Một lượt qua các ngày xếp hạng coin cho mọi bộ cùng lúc, bằng một phép nhân ma trận và một lần sắp xếp ổn định. Mọi book con nằm trong một tensor |
| Nhóm bộ trọng số chạy một lần, điểm cửa sổ lấy từ tổng cộng dồn | Không phải mô phỏng lại cửa sổ trượt nào cho nhóm này |
| Chỉ bước tinh chỉnh chạy lại cửa sổ | Mỗi ngày chỉ mô phỏng lại nhóm nhỏ ứng viên của nó |
| float32 khi tìm, float64 khi kiểm; chia lô | Nhanh trên GPU, giữ bộ nhớ vừa laptop, chính xác ở chỗ cần kiểm |
| Giữ bản chậm làm bản tham chiếu | Test kiểm hai cách cho cùng lãi lỗ từng ngày |

**Học được.** Tốc độ thay đổi chính cách làm, không chỉ thời gian chạy. Giờ việc tìm phủ cả không gian trọng số mỗi
ngày, và đoạn dài chạy được trên GPU. Kết quả vẫn trùng với bản tham chiếu chậm.

**Đã xây.** `qforecast/adaptive/engine.py`, và việc tìm chuyển sang dùng nó làm mặc định (`--backend numpy` giữ bản
tham chiếu).

---

## Hiện đang ở đâu

- C5 và book thích nghi đang chạy paper. Đó là test sạch duy nhất còn lại, vì mọi dữ liệu quá khứ đã dùng để thiết
  kế chúng.
- Trọng số thích nghi vẫn là công cụ nghiên cứu, cho tới khi dữ liệu chưa thấy cho thấy nó có ích.
- Mọi luật ở trên đều có test giữ: dữ liệu đúng thời điểm, không nhìn trước, luôn tính phí, lịch sử chỉ để loại.

## Bản đồ code

| Bước | Module | Vai trò |
|---|---|---|
| 1 | `qforecast/services/`, `core/`, `storage/` | Nhận dữ liệu realtime, feature tăng dần, Redis Streams / bus trong RAM |
| 1 | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` | Lịch sử perp đúng thời điểm và cube ngày |
| 1 | `qforecast/swing/features.py`, `signals.py` | Thống kê trượt, beta, xếp hạng chéo, thư viện tín hiệu |
| 2 | `qforecast/model/`, `trainer/` | Mô hình CNN-BiLSTM-attention, xuất ONNX |
| 2 | `qforecast/swing/book.py`, `strategy.py`, `paperbook.py`, `backtest.py` | Mô hình book, spec chiến thuật, book con, backtest kiểu sàn |
| 3 | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Kiểm định trung thực và vault test sạch |
| 3 | `artifacts/models/c5.json` | Spec của C5 |
| 4 | `qforecast/adaptive/search.py`, `weights.py` | Tìm trong không gian trọng số, trọng số thích nghi mỗi ngày |
| 4 | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` | Tài khoản ảo, bot paper, dashboard trên máy |
| 5 | `qforecast/adaptive/engine.py` | Engine chạy nhiều bộ trọng số cùng lúc |

## Chạy thử

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # dựng cube ngày
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # backtest kiểu sàn của C5
python -m qforecast.swing.paper --plan                              # lệnh C5 hôm nay (không giao dịch)
python -m qforecast.swing.dashboard                                 # sổ paper + lịch chạy mỗi ngày
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # trọng số thích nghi cho một đoạn ngày
pytest -q                                                           # chạy offline, dữ liệu giả
```

Trọng số của C5 được chọn trên chính đoạn lịch sử mà cube chứa. Backtest trên đó cho thấy code chạy đúng, không cho
biết nên kỳ vọng gì.

## Nhật ký xây dựng

Mỗi bước là một nhánh, có pull request nối lên nhánh trước.

| Bước | Ngày | Nhánh |
|---|---|---|
| 1. Dữ liệu | 2026-11-10 → 11-12 | `step-01/data-pipeline`, `step-02/signal-features` |
| 2. Mô hình | 2026-11-13 → 11-15 | `step-03/book-model` |
| 3. Thử nghiệm | 2026-11-16 → 11-19 | `step-04/research-toolkit`, `step-05/strategy-c5` |
| 4. Thích nghi | 2026-11-20 → 11-22 | `step-06/adaptive-weights`, `step-07/paper-dashboard` |
| 5. Tốc độ | 2026-11-23 → 11-26 | `step-08/fast-adaptive-engine` |
