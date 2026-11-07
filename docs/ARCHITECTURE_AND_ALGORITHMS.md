# Kiến trúc Hệ thống & Chi tiết Thuật toán (Architecture & Algorithms)

Tài liệu này trình bày thiết kế kiến trúc phần mềm, cơ sở lý thuyết và chi tiết hiện thực hóa hai thuật toán tìm kiếm đối kháng (**Alpha-Beta Pruning** và **Monte Carlo Tree Search - MCTS**), hệ thống hàm đánh giá thế cờ (**Evaluation Registry**) cùng giao diện đồ họa (**Tkinter GUI**).

---

## 1. Kiến trúc Tổng thể (System Architecture)

Hệ thống được thiết kế phân tầng theo nguyên lý tách biệt trách nhiệm (Separation of Concerns), đảm bảo các thuật toán AI hoàn toàn độc lập với giao diện người dùng và công cụ benchmark:

```text
+-------------------------------------------------------------------+
|                     main.py (CLI & GUI Entrypoint)                |
+---------------------------------+---------------------------------+
                                  |
          +-----------------------+-----------------------+
          v                                               v
+---------------------------------+     +-----------------------------------+
|         gui/ (Tkinter UI)       |     |    experiments/ (Benchmarking)    |
|  - app.py (Event & Thread Loop) |     |  - benchmark.py (Single Match)    |
|  - board_view.py (8x8 Canvas)   |     |  - tournament.py (Paired Swaps)   |
|  - eval_bar.py (Advantage Bar)  |     |  - run_parity_tournament.py       |
|  - agent_factory.py, theme.py   |     +-----------------+-----------------+
+-----------------+---------------+                       |
                  |                                       |
                  +-------------------+-------------------+
                                      v
            +---------------------------------------------------+
            |            chess_system/ (Core Contract)          |
            |  - game_controller.py (GameController, GameResult)|
            |  - ai_interface.py (ChessAI Protocol)             |
            +-------------------------+-------------------------+
                                      |
          +---------------------------+---------------------------+
          v                                                       v
+-----------------------------------+   +-----------------------------------+
|       algorithms/alpha_beta.py    |   |          algorithms/mcts.py       |
|  - AlphaBetaAI                    |   |  - MCTS, MCTSNode (mcts_node.py)  |
|  - MVV-LVA Move Ordering          |   |  - Guided Rollout (mcts_rollout)  |
+-----------------+-----------------+   +-----------------+-----------------+
                  |                                       |
                  +-------------------+-------------------+
                                      v
            +---------------------------------------------------+
            |        algorithms/eval_registry.py (EvalMode)     |
            |  - AB_CLASSIC (evaluation.py: Material + PST)     |
            |  - MCTS_PE (mcts_rollout.py: PeSTO Tapered + SEE) |
            +---------------------------------------------------+
```

### Giao ước tích hợp AI (`ChessAI` Protocol)
Mọi thuật toán trong `algorithms/` đều tuân thủ giao thức `ChessAI` (`chess_system/ai_interface.py`):
- **`choose_move(board: chess.Board) -> chess.Move`**: Nhận trạng thái bàn cờ, trả về một nước đi hợp lệ và **bảo toàn nguyên vẹn** trạng thái bàn cờ đầu vào (FEN và `move_stack`).
- **`abort_search() -> None`**: Dừng sớm quá trình tìm kiếm đang chạy khi người dùng nhấn `Pause`, `Undo`, `New Game` hoặc đóng cửa sổ trên GUI.
- **`last_search_stats: dict[str, int]`**: Lưu trữ thống kê của lượt tìm kiếm gần nhất (`depth`, `nodes`, `cutoffs` đối với Alpha-Beta; `iterations`, `rollouts` đối với MCTS).

---

## 2. Thuật toán Alpha-Beta Pruning (`algorithms/alpha_beta.py`)

### 2.1. Nguyên lý Tìm kiếm Minimax Giới hạn Độ sâu
Alpha-Beta thực hiện tìm kiếm cây trò chơi theo chiều sâu (Depth-First Search) đến độ sâu giới hạn $d \ge 1$ (tính theo nửa nước đi - *ply*). Điểm số toàn cục sử dụng góc nhìn của quân **Trắng**:
- **Quân Trắng (Maximizing player)**: Tối đa hóa điểm đánh giá, cập nhật cận dưới $\alpha$.
- **Quân Đen (Minimizing player)**: Tối thiểu hóa điểm đánh giá, cập nhật cận trên $\beta$.
- **Điều kiện cắt tỉa (Pruning condition)**: Khi $\alpha \ge \beta$, nhánh con hiện tại chắc chắn không thể cải thiện kết quả cho bên đang đi so với lựa chọn tốt hơn đã tìm thấy trước đó; thuật toán lập tức dừng duyệt nhánh này và tăng bộ đếm `cutoffs`.

### 2.2. Sắp xếp Nước đi Cộng dồn (`move_priority` & `order_moves`)
Hiệu quả cắt tỉa của Alpha-Beta phụ thuộc trực tiếp vào việc duyệt các nước đi mạnh trước. Hàm `move_priority(board, move)` gán điểm ưu tiên **cộng dồn** cho từng nước đi hợp lệ dựa trên 3 yếu tố chiến thuật:
1. **Bắt quân (Captures - MVV-LVA)**: Nước đi ăn quân được cộng `+10_000 + 10 * victim_value - attacker_value` (ưu tiên dùng quân giá trị thấp ăn quân giá trị cao của đối phương, tính cả bắt tốt qua đường *en passant*).
2. **Phong cấp Tốt (Promotions)**: Nước đi phong cấp được cộng thêm `+8_000 + PIECE_VALUES[promotion]`. Một nước vừa ăn quân vừa phong cấp sẽ nhận **cả hai mức thưởng**.
3. **Chiếu Vua (Checks)**: Nước đi trực tiếp chiếu tướng đối phương được cộng thêm `+50`.

---

## 3. Thuật toán Monte Carlo Tree Search (`algorithms/mcts.py`)

### 3.1. Chu trình 4 Pha với Công thức UCT
Với ngân sách $N$ vòng lặp (`iterations`), mỗi vòng lặp MCTS thực hiện 4 bước trên cây tìm kiếm có gốc là trạng thái hiện tại:
1. **Selection (Chọn nút)**: Từ nút gốc, đi xuống các nút con đã mở rộng hết (`not node.untried_moves`) theo chỉ số **UCT (Upper Confidence Bound 1 applied to Trees)**:
   $$\text{UCT}(v) = \bar{X}_v + c \sqrt{\frac{\ln N_{\text{parent}}}{N_v}}$$
   Trong đó $\bar{X}_v \in [-1, 1]$ là phần thưởng trung bình theo góc nhìn của người vừa thực hiện nước đi dẫn tới nút $v$, và hằng số khám phá mặc định $c = 2\sqrt{2}$.
2. **Expansion (Mở rộng)**: Tạo một nút con mới từ danh sách `untried_moves`. Tại nút gốc, các nước đi đã được lọc bỏ nước dẫn tới bị chiếu hết sau 1 nước (nếu còn nước an toàn khác) và sắp xếp theo điểm đánh giá tĩnh kèm SEE.
3. **Simulation / Rollout (Mô phỏng có hướng)**: Chạy mô phỏng tối đa `rollout_max_plies` nửa nước đi (mặc định 80 khi khởi tạo trực tiếp `MCTS`, và 24 khi khởi tạo qua GUI/CLI `build_agent` để đảm bảo tốc độ phản hồi thời gian thực).
   - Thay vì chọn ngẫu nhiên thuần túy (vốn dẫn tới các ván cờ vô nghĩa trong cờ vua), `rollout_move` ưu tiên chiếu hết ngay lập tức, sau đó lấy mẫu có trọng số (90% theo trọng số chiến thuật ăn quân có lợi qua SEE / phong cấp / chiếu / kiểm soát trung tâm, 10% ngẫu nhiên đồng đều).
   - Nếu hết giới hạn `rollout_max_plies` mà ván cờ chưa kết thúc, giá trị thế cờ tại lá được ước lượng qua hàm đánh giá và chuẩn hóa về $[-0.98, +0.98]$ bằng hàm $\tanh(\text{cp} / 600)$.
4. **Backpropagation (Lan truyền ngược)**: Cập nhật số lượt thăm (`visits += 1`) và cộng dồn phần thưởng (`value`) từ nút vừa mở rộng ngược về nút gốc, tự động đảo dấu theo lượt người chơi tại mỗi tầng (`node.player_just_moved`).

---

## 4. Hệ thống Đánh giá Thế cờ (`algorithms/eval_registry.py`)

Để đảm bảo tính công bằng khoa học khi so sánh hai thuật toán (**Evaluation Parity**), hệ thống cung cấp 2 chế độ đánh giá trong `EvalMode` có thể hoán đổi cho cả Alpha-Beta, MCTS và thanh Eval Bar:

| Chế độ (`EvalMode`) | Mô tả chi tiết | Nguồn cài đặt |
| :--- | :--- | :--- |
| **`ab_classic`** | **Đánh giá cổ điển**: Tổng giá trị chất (Material: Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900) cộng bảng vị trí quân (Piece-Square Tables). Bảng Vua tự động chuyển từ Middlegame sang Endgame khi không còn Hậu hoặc tổng chất phi-Tốt $\le 1300$. Điểm chiếu hết ưu tiên đường ngắn nhất: $\pm(\text{MATE\_SCORE} - \text{ply})$. | `algorithms/evaluation.py` |
| **`mcts_pe`** | **Đánh giá PeSTO Tapered + SEE**: Nội suy mượt (continuous phase interpolation) giữa điểm Middlegame và Endgame dựa trên tổng chất phi-Tốt hiện tại trên bàn cờ. Kết hợp đánh giá trao đổi tĩnh **SEE (`exchange_gain`)** để phát hiện các quân đang bị treo hoặc chuỗi ăn quân trả đũa. | `algorithms/mcts_rollout.py` |

---

## 5. Thiết kế Giao diện Đồ họa (`gui/`)

Giao diện Tkinter được phân rã thành 5 module độc lập:
1. **`gui/theme.py`**: Quản lý bảng màu Dark Mode, phông chữ, kích thước chuẩn và bộ ký tự Unicode quân cờ đặc ruột (`♚♛♜♝♞♟`) kết hợp kỹ thuật vẽ viền đa hướng giúp quân Trắng và Đen đều đặc ruột, sắc nét trên mọi màu ô cờ.
2. **`gui/board_view.py`**: Ánh xạ tọa độ ô cờ $\leftrightarrow$ pixel hỗ trợ lật bàn cờ (`Flip Board`), vẽ ô cờ, gợi ý nước đi hợp lệ (chấm tròn cho ô trống, vòng tròn viền cho ô bắt quân / *en passant*), hộp thoại chọn quân phong cấp dọc ngay trên bàn cờ và **huy hiệu kết thúc ván cờ bằng vector** (vương miện vàng trên Vua thắng, dấu chéo đỏ trên Vua thua, `½` khi hòa) không gây trễ font GDI.
3. **`gui/eval_bar.py`**: Tính toán tỷ lệ ưu thế theo hàm logistic sigmoid $\sigma(\text{cp} / 400) \in [0.05, 0.95]$ và vẽ thanh Evaluation Bar dọc bên trái bàn cờ, tự động đảo chiều khi lật bàn cờ.
4. **`gui/agent_factory.py`**: Kiểm tra hợp lệ dữ liệu nhập từ người dùng/CLI và khởi tạo thực thể `AlphaBetaAI` hoặc `MCTS`.
5. **`gui/app.py`**: Điều phối vòng lặp sự kiện, quản lý luồng nền (`self.ai_worker_thread`), đồng bộ kết quả qua `queue.Queue` và ngăn chặn triệt để xung đột đa luồng khi `Pause / Resume / Undo / New Game` bằng `abort_search()` kết hợp `join()`.
