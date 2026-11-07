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
- **Chế độ chơi:** Human vs AI, AI vs Human, AI vs AI (tự động phát, Pause/Resume với cơ chế dừng sớm worker thread an toàn, Step từng nước), Human vs Human.
- **Thanh đánh giá (Eval Bar):** Hiển thị ưu thế thế cờ theo thời gian thực (đơn vị centipawn / tốt), tự động đổi hướng theo bàn cờ (`Flip Board`), có thể Bật/Tắt hoặc đổi hàm đánh giá tức thì.
- **Nhận diện kết thúc ván cờ & Phong cấp:** Hiển thị huy hiệu vector (vương miện vàng cho Vua thắng, dấu chéo đỏ cho Vua thua, ký hiệu `½` khi hòa) và bảng chọn quân phong cấp (Hậu, Mã, Xe, Tượng) trực tiếp trên bàn cờ theo phong cách Chess.com.
- **Tùy biến tham số động:** Tự động hiển thị tham số phù hợp theo thuật toán (Depth cho Alpha-Beta, Sims cho MCTS, chuyển đổi Evaluation Mode).

---

### 2.2. Chạy Thực nghiệm Không Giao diện (Headless CLI)

#### Đấu một trận đơn (Match)
```powershell
python main.py match --white alpha_beta --white-depth 3 --black mcts --black-iterations 100 --max-plies 200 --output-dir experiments/results/demo_match
```

#### Đấu giải đổi bên (Tournament)
```powershell
python main.py tournament --depth 2 --iterations 100 --ab-eval-mode mcts_pe --eval-mode mcts_pe --rounds 5 --max-plies 500 --output-dir experiments/results/tourney_demo
```

#### Chạy Batch Thực nghiệm Đa tiến trình (Evaluation Parity)
```powershell
python experiments/run_parity_tournament.py --workers 4 --rounds 5 --max-plies 500 --output-dir experiments/results/parity_eval_batch
```
Kết quả giải đấu được tự động xuất ra các tệp `.json`, `.csv`, `.pgn`, `standings.csv` từng cặp và tổng hợp vào `master_standings.csv`.

---

## 3. Cấu trúc Dự án

```text
Chess_Engine/
├── main.py                              # Entrypoint điều khiển GUI và CLI runner
├── requirements.txt                     # Thư viện phụ thuộc (python-chess)
├── README.md                            # Tài liệu hướng dẫn sử dụng
├── chess_system/                        # Quản lý trạng thái và luật cờ vua
│   ├── ai_interface.py                  # Protocol giao tiếp ChessAI
│   └── game_controller.py               # Bộ điều khiển ván cờ, FIDE rules, claim_draw
├── algorithms/                          # Thuật toán tìm kiếm và đánh giá
│   ├── alpha_beta.py                    # Alpha-Beta Pruning với sắp xếp nước đi MVV-LVA
│   ├── evaluation.py                    # Đánh giá tĩnh cổ điển (Material + PST)
│   ├── mcts.py                          # MCTS engine (UCT selection, rollout, backprop)
│   ├── mcts_node.py                     # Cấu trúc nút cây tìm kiếm MCTS
│   ├── mcts_rollout.py                  # Chính sách rollout và PeSTO tapered evaluation
│   └── eval_registry.py                 # Quản lý chuyển đổi chế độ đánh giá (EvalMode)
├── gui/                                 # Giao diện đồ họa Tkinter (kiến trúc module hóa)
│   ├── app.py                           # Bộ điều phối sự kiện, vòng lặp ván cờ và luồng AI
│   ├── board_view.py                    # Hệ tọa độ, vẽ bàn cờ 8x8, huy hiệu Vua & phong cấp
│   ├── eval_bar.py                      # Tính toán điểm ưu thế và vẽ thanh Evaluation Bar
│   ├── agent_factory.py                 # Khởi tạo AI agent, kiểm tra tham số & thống kê
│   ├── theme.py                         # Cấu hình bảng màu, phông chữ, biểu tượng & dark theme
│   └── assets/                          # Tài nguyên đồ họa bổ trợ
├── experiments/                         # Hạ tầng thực nghiệm và đối đầu
│   ├── benchmark.py                     # Chạy và lưu kết quả trận đấu đơn
│   ├── benchmark_alpha_beta.py          # So sánh hiệu năng cắt tỉa Alpha-Beta có/không sắp xếp
│   ├── tournament.py                    # Giải đấu ghép cặp đổi bên (color-swapped, độc lập seed)
│   ├── run_parity_tournament.py         # Chạy giải đấu đối đầu song song đa tiến trình
│   └── results/                         # Thư mục lưu kết quả thực nghiệm
├── tests/                               # Bộ kiểm thử tự động (75 tests)
│   ├── test_game.py                     # Kiểm thử GameController và luật cờ
│   ├── test_benchmark.py                # Kiểm thử benchmark và tournament
│   ├── test_alpha_beta.py               # Kiểm thử Alpha-Beta và hàm đánh giá tĩnh
│   ├── test_mcts.py                     # Kiểm thử MCTS, UCT node, rollout và giải thế cờ
│   └── test_integration.py              # Kiểm thử tích hợp GUI, AI, Eval Bar, hủy luồng & CLI
└── docs/                                # Tài liệu tham khảo dự án
```

---

## 4. Thuật toán & Chế độ Đánh giá (Evaluation Modes)

Hệ thống cho phép cấu hình độc lập hai cơ chế đánh giá thế cờ cho từng thuật toán và thanh Eval Bar:
- **`ab_classic`**: Đánh giá tĩnh truyền thống dựa trên chênh lệch giá trị quân cờ (Material) và bảng điểm vị trí quân (Piece-Square Tables).
- **`mcts_pe`**: Đánh giá PeSTO nâng cao với nội suy giai đoạn ván cờ (Tapered Evaluation giữa Middlegame/Endgame), bảng vị trí đối xứng và Static Exchange Evaluation (SEE).

Tất cả AI tuân thủ giao thức `choose_move(board: chess.Board) -> chess.Move`, hỗ trợ dừng tìm kiếm sớm qua `abort_search()`, bảo toàn trạng thái bàn cờ gốc và xuất chỉ số tìm kiếm (`last_search_stats`).
