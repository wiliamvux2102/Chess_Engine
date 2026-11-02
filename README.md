# Chess Engine (Alpha-Beta & MCTS)

Đồ án cờ vua AI hoàn chỉnh tích hợp thuật toán Alpha-Beta Pruning và Monte Carlo Tree Search (MCTS), giao diện đồ họa Tkinter (Human vs AI, Live AI vs AI, thanh Eval Bar phong cách Chess.com) cùng bộ công cụ thực nghiệm benchmark/tournament headless.
---

## 1. Cài đặt và Kiểm thử

Yêu cầu Python 3.10 trở lên. Trong terminal/PowerShell tại thư mục dự án:

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Chạy toàn bộ test suite (unit & integration tests)
python -m unittest discover -s tests -v
```

---

## 2. Hướng dẫn sử dụng

### 2.1. Khởi chạy Giao diện (GUI)

```powershell
# Khởi chạy GUI mặc định (Human vs AI)
python main.py

# Xem trực tiếp AI vs AI (Alpha-Beta depth 3 vs MCTS 100 simulations, trễ 300ms)
python main.py --white alpha_beta --white-depth 3 --black mcts --black-iterations 100 --delay 300

# Khởi chạy với cấu hình Evaluation Mode cụ thể
python main.py --white alpha_beta --white-eval-mode mcts_pe --black mcts --black-eval-mode mcts_pe
```

**Tính năng nổi bật trên GUI:**
- **Chế độ chơi:** Human vs AI, AI vs Human, AI vs AI (tự động phát, Pause/Resume, Step từng nước), Human vs Human.
- **Thanh đánh giá (Eval Bar):** Hiển thị ưu thế thế cờ theo thời gian thực (đơn vị centipawn / tốt), tự động đổi hướng theo bàn cờ (`Flip Board`), có thể Bật/Tắt tức thì.
- **Phong cấp trực quan:** Bảng chọn quân phong cấp (Hậu, Xe, Tượng, Mã) hiển thị ngay trên bàn cờ.
- **Tùy biến tham số:** Thay đổi động theo thuật toán (Depth cho Alpha-Beta, Iterations cho MCTS, chuyển đổi Evaluation Mode).

---

### 2.2. Chạy Thực nghiệm Không Giao diện (Headless CLI)

#### Đấu một trận đơn (Match)
```powershell
python main.py match --white alpha_beta --white-depth 3 --black mcts --black-iterations 100 --max-plies 200 --output-dir experiments/results/demo_match
```

#### Đấu giải đổi bên (Tournament)
```powershell
python main.py tournament --depth 2 --iterations 100 --rounds 5 --max-plies 500 --output-dir experiments/results/tourney_demo
```
Kết quả giải đấu được tự động xuất ra các tệp `.json`, `.csv`, `.pgn` và `standings.csv` tổng hợp.

---

## 3. Cấu trúc Dự án

```text
Chess_Engine/
├── main.py                     # Entrypoint điều khiển GUI và CLI runner
├── requirements.txt            # Thư viện phụ thuộc (python-chess)
├── README.md                   # Tài liệu hướng dẫn sử dụng
├── chess_system/               # Quản lý trạng thái và luật cờ vua
│   ├── ai_interface.py         # Protocol giao tiếp ChessAI
│   └── game_controller.py      # Bộ điều khiển ván cờ, FIDE rules, claim_draw
├── algorithms/                 # Thuật toán tìm kiếm và đánh giá
│   ├── alpha_beta.py           # Alpha-Beta Pruning với sắp xếp nước đi MVV-LVA
│   ├── evaluation.py           # Đánh giá tĩnh cổ điển (Material + PST)
│   ├── mcts.py                 # MCTS engine (UCT selection, rollout, backprop)
│   ├── mcts_node.py            # Cấu trúc nút cây tìm kiếm MCTS
│   ├── mcts_rollout.py         # Rollout chính sách và PeSTO tapered evaluation
│   └── eval_registry.py        # Quản lý chuyển đổi chế độ đánh giá
├── gui/                        # Giao diện đồ họa Tkinter
│   ├── app.py                  # Canvas 8x8, Eval Bar, Live AI vs AI, Promotion dialog
│   └── assets/                 # Tài nguyên đồ họa bổ trợ
├── experiments/                # Hạ tầng thực nghiệm và đối đầu
│   ├── benchmark.py            # Chạy và lưu trận đấu đơn
│   ├── tournament.py           # Giải đấu ghép cặp đổi bên (color-swapped)
│   └── results/                # Thư mục lưu kết quả thực nghiệm
├── tests/                      # Bộ kiểm thử tự động
│   ├── test_game.py            # Kiểm thử GameController và luật cờ
│   ├── test_benchmark.py       # Kiểm thử benchmark và tournament
│   ├── test_alpha_beta.py      # Kiểm thử Alpha-Beta
│   ├── test_mcts.py            # Kiểm thử MCTS
│   └── test_integration.py     # Kiểm thử tích hợp GUI, AI, Eval Bar, CLI
└── docs/                       # Tài liệu tham khảo dự án
```

---

## 4. Thuật toán & Chế độ Đánh giá (Evaluation Modes)

Hệ thống cho phép cấu hình độc lập hai cơ chế đánh giá thế cờ:
- **`ab_classic`**: Đánh giá tĩnh truyền thống dựa trên chênh lệch giá trị quân cờ (Material) và bảng điểm vị trí quân (Piece-Square Tables).
- **`mcts_pe`**: Đánh giá PeSTO nâng cao với nội suy giai đoạn ván cờ (Tapered Evaluation giữa Opening/Endgame), bảng vị trí nâng cao và Static Exchange Evaluation (SEE).

Tất cả AI tuân thủ giao thức `choose_move(board: chess.Board) -> chess.Move`, không làm thay đổi bàn cờ gốc và hỗ trợ xuất chỉ số tìm kiếm (`last_search_stats`).
