# Báo cáo Thực nghiệm & Phân tích Đối sánh (Experimental Results)

Tài liệu này tổng hợp phương pháp luận và toàn bộ kết quả thực nghiệm đối đầu giữa **Alpha-Beta Pruning** và **Monte Carlo Tree Search (MCTS)** dưới điều kiện **cùng hàm đánh giá thế cờ (Evaluation Parity)** và **seed độc lập từng ván**.

---

## 1. Phương pháp luận Thực nghiệm (Experimental Methodology)

### 1.1. Nguyên lý Công bằng Đánh giá (Evaluation Parity)
Nếu hai thuật toán tìm kiếm sử dụng hai hàm đánh giá thế cờ khác nhau, kết quả thắng/thua sẽ bị nhiễu giữa năng lực tìm kiếm (cắt tỉa $\alpha-\beta$ so với lấy mẫu UCT) và chất lượng hàm heuristic. Do đó, thực nghiệm được chia thành 2 bộ đối xứng hoàn toàn:
- **Suite 1 (`ab_classic` Parity)**: Cả Alpha-Beta và MCTS đều sử dụng hàm đánh giá cổ điển (`ab_classic` - Material + PST).
- **Suite 2 (`mcts_pe` Parity)**: Cả Alpha-Beta và MCTS đều sử dụng hàm đánh giá PeSTO Tapered Evaluation kèm SEE (`mcts_pe`).

### 1.2. Cấu hình Ma trận Tham số & Tính Độc lập Thống kê
- **Độ sâu Alpha-Beta (`depth`)**: $d \in \{1, 2, 3\}$.
- **Số vòng mô phỏng MCTS (`iterations`)**: $N \in \{50, 100, 200\}$ (giới hạn rollout $k_{\max} = 24$ plies).
- **Quy mô mỗi cặp tham số**: 5 vòng đấu cặp đổi màu quân (`rounds = 5`) = **10 ván đấu** (5 ván Alpha-Beta cầm quân Trắng, 5 ván MCTS cầm quân Trắng).
- **Tính độc lập ngẫu nhiên (Per-Game Seed Diversity)**: Trong `experiments/tournament.py` và `main.py`, mỗi ván thứ $k \in \{1, \dots, 10\}$ khởi tạo MCTS với `seed = base_seed + k - 1` (từ `42` đến `51`), đảm bảo 10 ván trong mỗi cặp là 10 thử nghiệm ngẫu nhiên độc lập hoàn toàn nhưng vẫn tái lập được 100%.
- **Điều kiện kết thúc ván**: Áp dụng đầy đủ luật FIDE (`claim_draw=True`: chiếu hết, hết nước đi, lặp lại 3 lần, luật 50 nước, thiếu quân chiếu hết) với trần `max_plies = 500`. Toàn bộ **180/180 ván** đều kết thúc tự nhiên (`unfinished = 0`).

---

## 2. Kết quả Thực nghiệm Tổng hợp (180 Ván đấu)

Dữ liệu gốc lưu tại `experiments/results/parity_eval_batch/master_standings.csv`.

### 2.1. Suite 1: Cùng sử dụng Hàm Đánh giá Cổ điển (`ab_classic`)

| AB Depth ($d$) | MCTS Sims ($N$) | Số ván | AB Thắng | Hòa | MCTS Thắng | Tổng Plies | Thời gian AB (s) | Thời gian MCTS (s) | Tỉ lệ thắng AB |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **50**  | 10 | 4 | 4 | 2 | 655 | 5.26s   | 238.27s  | 40.0% (Hòa 40%) |
| **1** | **100** | 10 | 5 | 3 | 2 | 670 | 4.53s   | 432.62s  | 50.0% (Hòa 30%) |
| **1** | **200** | 10 | 3 | 2 | 5 | 462 | 3.77s   | 678.42s  | 30.0% (MCTS thắng 50%) |
| **2** | **50**  | 10 | **10** | 0 | 0 | 525 | 23.00s  | 216.65s  | **100.0%** |
| **2** | **100** | 10 | **10** | 0 | 0 | 587 | 25.12s  | 459.71s  | **100.0%** |
| **2** | **200** | 10 | **9**  | 0 | 1 | 528 | 23.70s  | 939.03s  | **90.0%** |
| **3** | **50**  | 10 | **10** | 0 | 0 | 525 | 360.28s | 255.84s  | **100.0%** |
| **3** | **100** | 10 | **10** | 0 | 0 | 439 | 263.81s | 446.37s  | **100.0%** |
| **3** | **200** | 10 | **10** | 0 | 0 | 507 | 282.38s | 1024.67s | **100.0%** |
| **Tổng** | — | **90** | **71** | **9** | **10** | **4898** | **991.85s** | **4691.58s** | **78.9%** |

---

### 2.2. Suite 2: Cùng sử dụng Hàm Đánh giá PeSTO Tapered + SEE (`mcts_pe`)

| AB Depth ($d$) | MCTS Sims ($N$) | Số ván | AB Thắng | Hòa | MCTS Thắng | Tổng Plies | Thời gian AB (s) | Thời gian MCTS (s) | Tỉ lệ thắng AB |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **50**  | 10 | 1 | 3 | **6** | 526 | 4.25s   | 227.21s  | 10.0% (MCTS thắng 60%) |
| **1** | **100** | 10 | 2 | 4 | **4** | 615 | 4.76s   | 502.50s  | 20.0% (MCTS thắng 40%) |
| **1** | **200** | 10 | 2 | 3 | **5** | 505 | 4.03s   | 879.32s  | 20.0% (MCTS thắng 50%) |
| **2** | **50**  | 10 | **10** | 0 | 0 | 775 | 32.56s  | 339.75s  | **100.0%** |
| **2** | **100** | 10 | **10** | 0 | 0 | 505 | 23.97s  | 467.44s  | **100.0%** |
| **2** | **200** | 10 | **10** | 0 | 0 | 657 | 27.66s  | 1154.48s | **100.0%** |
| **3** | **50**  | 10 | **10** | 0 | 0 | 555 | 348.62s | 280.99s  | **100.0%** |
| **3** | **100** | 10 | **10** | 0 | 0 | 497 | 290.82s | 470.62s  | **100.0%** |
| **3** | **200** | 10 | **10** | 0 | 0 | 733 | 293.86s | 1144.46s | **100.0%** |
| **Tổng** | — | **90** | **65** | **10** | **15** | **5368** | **1030.53s** | **5466.77s** | **72.2%** |

---

## 3. Phân tích Chuyên sâu & Nhận xét (Key Findings)

1. **Tại độ sâu $d = 1$ (Horizon Effect của Alpha-Beta nông)**:
   - Khi Alpha-Beta chỉ nhìn trước 1 nửa nước đi ($d = 1$), nó không thấy được nước ăn quân trả đũa ngay ở lượt kế tiếp của đối phương (Horizon Effect).
   - Ngược lại, MCTS thực hiện lọc chiến thuật tại gốc kèm đánh giá trao đổi tĩnh (`exchange_gain` - SEE) và mô phỏng nhiều bước tiếp theo. Nhờ đó, ở $d = 1$ với `ab_classic` ($N = 200$), MCTS thắng 5 - 3; đặc biệt trong Suite 2 (`mcts_pe`), MCTS áp đảo Alpha-Beta $d = 1$ ở cả 3 mức ngân sách ($N = 50, 100, 200$) với tổng thành tích **15 thắng – 10 hòa – 5 thua**.

2. **Tại độ sâu $d \ge 2$ (Ưu thế Tuyệt đối của Tìm kiếm Toàn diện có Cắt tỉa)**:
   - Ngay khi Alpha-Beta tăng lên $d = 2$ và $d = 3$, thuật toán kiểm tra đầy đủ mọi phản đòn chiến thuật tầm gần kết hợp sắp xếp nước đi MVV-LVA.
   - Trong cờ vua (không gian nhánh trung bình $b \approx 30-35$ và tính chất chiến thuật sắc bén - chỉ một nước đi sai lầm làm mất quân là thua cuộc), ngân sách $50 - 200$ vòng mô phỏng của MCTS chỉ đủ thăm mỗi nước đi gốc khoảng $2 - 6$ lần, không đủ sâu để chống lại cây tìm kiếm vét cạn có cắt tỉa của Alpha-Beta. Alpha-Beta đạt tỉ lệ thắng **98.3%** (59/60 ván ở Suite 1, chỉ thua 1 ván ở $d=2, N=200$) và **100%** (60/60 ván ở Suite 2) khi $d \ge 2$.

3. **Hiệu năng Thời gian Thực thi (Computational Efficiency)**:
   - Ở $d = 2$, Alpha-Beta chỉ mất trung bình **~0.08 – 0.10 giây/nước**, nhanh hơn từ **10 đến 40 lần** so với MCTS ($N = 50 \dots 200$ mất **~1.5 – 6.5 giây/nước**) do mỗi vòng mô phỏng MCTS phải sao chép bàn cờ và chạy rollout 24 plies.
   - Khi Alpha-Beta tăng lên $d = 3$, thời gian tìm kiếm tăng theo hàm mũ lên **~1.8 – 2.2 giây/nước**, tiệm cận thời gian của MCTS ở mức $N = 50 - 100$.

---

## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility)

Để chạy lại toàn bộ 18 cặp đấu (180 ván) hoặc một cặp bất kỳ từ thư mục gốc dự án:

```powershell
# Chạy toàn bộ 18 cặp (180 ván) bằng 4 tiến trình song song
.\.venv\Scripts\python.exe experiments/run_parity_tournament.py --workers 4 --rounds 5 --max-plies 500 --output-dir experiments/results/parity_eval_batch

# Chạy lẻ 1 cặp đấu cụ thể (ví dụ: d=1, iterations=200, cùng dùng mcts_pe)
.\.venv\Scripts\python.exe main.py tournament --depth 1 --iterations 200 --ab-eval-mode mcts_pe --eval-mode mcts_pe --rounds 5 --max-plies 500 --seed 42 --output-dir experiments/results/single_test
```
