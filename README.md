# HW2 Chess

Khung bài tập cờ vua của nhóm 4 thành viên. Phần của Chính đã triển khai:
quản lý ván cờ, hợp đồng AI, benchmark và tournament. Thuật toán và GUI để
file rỗng cho các thành viên phụ trách. Chưa có chương trình chơi cờ hoàn chỉnh.

## Cài đặt và kiểm thử

Yêu cầu Python 3.10 trở lên. Trong PowerShell, tại thư mục `HW2_Chess`:

```powershell
$env:PYTHONUTF8 = "1"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Việc sử dụng python-chess cho luật cờ cần được giảng viên chấp thuận theo tài liệu
phân công. Không sử dụng engine ngoài.

## Cấu trúc và người phụ trách

```text
HW2_Chess/
├── main.py                         # QHuy, rỗng
├── requirements.txt                # Chính
├── README.md                       # Hướng dẫn hạ tầng của Chính
├── chess_system/                   # Chính, đã triển khai
│   ├── __init__.py
│   ├── game_controller.py
│   └── ai_interface.py
├── algorithms/
│   ├── __init__.py                 # Rỗng
│   ├── alpha_beta.py               # Hải, rỗng
│   ├── evaluation.py               # Hải, rỗng
│   ├── mcts.py                     # Thạch, rỗng
│   └── mcts_node.py                # Thạch, rỗng
├── gui/                            # QHuy, file rỗng
│   ├── __init__.py
│   ├── app.py
│   └── assets/
├── experiments/                    # Chính, đã triển khai
│   ├── __init__.py
│   ├── benchmark.py
│   ├── tournament.py
│   └── results/
├── tests/
│   ├── __init__.py
│   ├── test_game.py                # Chính
│   ├── test_benchmark.py           # Chính
│   ├── test_alpha_beta.py          # Hải, rỗng
│   ├── test_mcts.py                # Thạch, rỗng
│   └── test_integration.py         # QHuy, rỗng
└── docs/                           # Nơi đặt report.pdf sau khi viết báo cáo
```

Không tạo PDF rỗng vì đó không phải một tài liệu PDF hợp lệ.

## GameController

```python
from chess_system.game_controller import GameController

game = GameController()
game.push("e2e4")              # Hoặc truyền chess.Move
print(game.turn)              # chess.WHITE / chess.BLACK
print(game.legal_moves)       # Tuple nước đi hợp lệ
print(game.result())          # GameResult(result, winner, termination)
game.pop()                    # Hoàn tác một nước
```

`game.board` trả về bản sao có lịch sử, có thể đưa cho AI mà không làm thay đổi
bàn cờ controller. `push()` kiểm tra nước đi hợp lệ, bao gồm nhập thành,
en passant, phong cấp (`a7a8q`) và từ chối đi tiếp sau khi kết thúc ván.
`pop()` trên ván chưa có nước đi phát sinh `IndexError`.

Kết quả thống nhất: `1-0`, `0-1`, `1/2-1/2`, `*` (chưa kết thúc).
Mặc định `claim_draw=True`: tự chấp nhận hòa khi đủ điều kiện yêu cầu hòa
theo luật python-chess, kể cả khi có thể yêu cầu bằng nước đi tiếp theo.
Truyền `claim_draw=False` nếu chỉ muốn áp dụng hòa tự động bắt buộc.
GUI và các thuật toán cần dùng cùng chính sách này khi kiểm tra terminal state.

## Hợp đồng cho Hải và Thạch

Mỗi AI cung cấp phương thức:

```python
def choose_move(self, board: chess.Board) -> chess.Move:
    ...
```

- Trả về một `chess.Move` hợp lệ. Runner không gọi AI trên thế cờ kết thúc.
- Giữ nguyên FEN và lịch sử của board nhận vào khi trả về. Dùng `board.copy()`
  hoặc cân bằng mọi `push()` bằng `pop()`, kể cả khi thoát sớm.
- AI tự xử lý search depth, iterations hoặc time budget qua constructor.
- Tùy chọn cập nhật `self.last_search_stats` sau mỗi lần tìm kiếm:
  `{"depth": 3, "nodes": 1200}` cho Alpha-Beta hoặc
  `{"iterations": 1000, "rollouts": 1000}` cho MCTS.
  Mỗi metric là số nguyên không âm; metric không cung cấp ghi thành ô trống/null.
  Không gộp node count Alpha-Beta với iterations/rollouts MCTS.
- Với AI ngẫu nhiên, dùng bộ sinh ngẫu nhiên riêng và cấu hình seed để tái lập.

`ChessAI` là Protocol để chú thích kiểu; không bắt buộc kế thừa.

## Chạy benchmark và tournament sau khi có hai AI

Các tên class và tham số dưới đây là ví dụ hợp đồng tích hợp. Hải và Thạch cần
thống nhất tên thực tế; hiện các file thuật toán còn rỗng nên đoạn này chưa chạy được.
Chạy script tích hợp tại thư mục gốc `HW2_Chess` để import đúng package.

```python
from algorithms.alpha_beta import AlphaBetaAI
from algorithms.mcts import MCTSAI
from experiments.tournament import run_tournament, summarize

matches = run_tournament(
    factory_a=lambda: AlphaBetaAI(depth=3),
    factory_b=lambda: MCTSAI(iterations=1000, seed=42),
    name_a="Alpha-Beta",
    name_b="MCTS",
    rounds=2,
    max_plies=300,
    config={
        "Alpha-Beta": {"depth": 3},
        "MCTS": {"iterations": 1000, "seed": 42},
    },
    output_dir="experiments/results/run_001",
)
print(summarize(matches))
```

Một opening × hai rounds × hai lượt đổi màu = bốn ván. Có thể truyền
`openings=[fen1, fen2, ...]` để dùng nhiều thế cờ. Mỗi ván tạo AI mới từ factory.
`config` là metadata do người gọi ghi, không tự thay đổi ngân sách của AI.
Để so sánh công bằng, xác định ngân sách trước và ghi rõ cách so sánh trong báo cáo.

Mỗi ván xuất `.json` (kết quả, cấu hình và từng nước), `.csv` (thời gian, metric
từng nước), `.pgn` (phát lại ván cờ); `standings.csv` tổng hợp thắng/hòa/thua,
chưa hoàn tất, số nước từng AI và tổng thời gian suy nghĩ. Thời gian dùng
`perf_counter`, chỉ đo lời gọi `choose_move`, không gồm xuất file hay vẽ GUI.
`plies` tính theo nửa lượt: mỗi lần một bên đi là một ply.

`max_plies` giới hạn độ dài ván, không giới hạn thời gian thực thi AI.
Ván chạm giới hạn có kết quả `*`, termination `MOVE_LIMIT`, được đếm riêng
ở `unfinished`. AI lỗi, sửa input hoặc trả nước bất hợp lệ sẽ làm runner báo lỗi
để sửa; không tự chuyển thành thua. Kết quả từng ván được lưu ngay khi hoàn tất.
Thư mục output tournament phải chưa tồn tại để tránh ghi đè thí nghiệm cũ.

Muốn chạy riêng một ván, dùng `run_match(...)` và `save_match(match, folder, id)`
trong `experiments.benchmark`. Các AI giả chỉ có trong file test để kiểm tra
hạ tầng; không phải thuật toán dùng cho bài nộp.
