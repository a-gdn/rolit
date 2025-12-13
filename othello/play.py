import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import curses
import argparse
import os
import math
import time

# ==========================================
# 1. GAME LOGIC (ROLIT/OTHELLO)
# ==========================================
class RolitEnv:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        self.board[mid-1, mid-1] = 1
        self.board[mid, mid] = 1
        self.board[mid-1, mid] = 2
        self.board[mid, mid-1] = 2
        self.done = False
        self.move_count = 0 
        return self.board.copy()

    def get_legal_moves(self, player):
        moves = set()
        opponent = 2 if player == 1 else 1
        rows, cols = np.where(self.board == 0)
        for r, c in zip(rows, cols):
            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == opponent:
                    temp_r, temp_c = nr, nc
                    while 0 <= temp_r < self.board_size and 0 <= temp_c < self.board_size:
                        if self.board[temp_r, temp_c] == 0: break
                        if self.board[temp_r, temp_c] == player:
                            moves.add((r, c))
                            break
                        temp_r += dr
                        temp_c += dc
                    if (r,c) in moves: break 
        return list(moves)

    def flip_pieces(self, r, c, player):
        opponent = 2 if player == 1 else 1
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            to_flip = []
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent:
                    to_flip.append((nr, nc))
                elif self.board[nr, nc] == player:
                    for fx, fy in to_flip:
                        self.board[fx, fy] = player
                    break
                else: break
                nr += dr
                nc += dc

    def step(self, action, player):
        r, c = action
        self.board[r, c] = player
        self.flip_pieces(r, c, player)
        self.move_count += 1
        if not np.any(self.board == 0) or np.sum(self.board == 1) == 0 or np.sum(self.board == 2) == 0: 
            self.done = True
        return self.board.copy(), 0, self.done

    def winner(self):
        p1 = np.sum(self.board == 1)
        p2 = np.sum(self.board == 2)
        if p1 > p2: return 1
        if p2 > p1: return 2
        return 0

# ==========================================
# 2. NEURAL NETWORK HELPERS
# ==========================================
def preprocess_state(board, player):
    # Plane 0: My pieces, Plane 1: Opponent pieces, Plane 2: Turn count
    state = np.zeros((8, 8, 3), dtype=np.float32) 
    state[:, :, 0] = (board == player).astype(np.float32)
    state[:, :, 1] = ((board != player) & (board != 0)).astype(np.float32)
    state[:, :, 2] = np.sum(board != 0) / 64.0
    return state

# ==========================================
# 3. MCTS (AI THINKING)
# ==========================================
class MCTS:
    def __init__(self, env, model, simulations=400, cpuct=2.0):
        self.env = env
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        # Using dicts for Q, N, P to store tree data
        self.Q = {}
        self.N = {}
        self.P = {}

    def run(self, board, player, move_count):
        # Clear tree for fresh search
        self.Q = {}; self.N = {}; self.P = {}
        
        for _ in range(self.simulations):
            self._simulate(board.copy(), player)
            
        pi = np.zeros(self.env.board_size**2)
        root = (board.tobytes(), player)
        legal = self.env.get_legal_moves(player)
        
        # Calculate policy from visit counts
        for r, c in legal:
            action = r * self.env.board_size + c
            pi[action] = self.N.get((root, action), 0)
            
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            return pi / sum_pi
        return pi

    def _simulate(self, board, player):
        key = (board.tobytes(), player)
        temp_env = RolitEnv(self.env.board_size)
        temp_env.board = board
        legal = temp_env.get_legal_moves(player)

        # 1. Terminal Check
        if not legal:
            opp = 2 if player == 1 else 1
            if not temp_env.get_legal_moves(opp):
                w = temp_env.winner()
                if w == 0: return 0
                return 1 if w == player else -1
            return -self._simulate(board, opp)

        # 2. Expansion
        if key not in self.P:
            inp = preprocess_state(board, player)[np.newaxis, ...]
            # AlphaZero model returns (policy, value)
            policy, value = self.model.predict(inp, verbose=0)
            policy = policy[0]
            value = value[0][0]
            
            # Mask illegal moves
            mask = np.zeros_like(policy)
            for r, c in legal:
                mask[r * self.env.board_size + c] = 1
            
            policy = policy * mask
            if np.sum(policy) > 0:
                policy /= np.sum(policy)
            else:
                policy = mask / np.sum(mask)
                
            self.P[key] = policy
            return value

        # 3. Selection (PUCT)
        total_N = sum(self.N.get((key, r*8+c), 0) for r,c in legal)
        sqrt_total = math.sqrt(total_N + 1e-8)
        
        best_u = -float('inf')
        best_action = None
        
        for r, c in legal:
            a = r * 8 + c
            q = self.Q.get((key, a), 0)
            n = self.N.get((key, a), 0)
            p = self.P[key][a]
            u = q + self.cpuct * p * sqrt_total / (1 + n)
            
            if u > best_u:
                best_u = u
                best_action = a
        
        # 4. Step
        r, c = divmod(best_action, 8)
        next_board, _, _ = temp_env.step((r, c), player)
        next_player = 2 if player == 1 else 1
        
        val = self._simulate(next_board, next_player)
        
        # 5. Backup
        v = -val
        if (key, best_action) in self.Q:
            self.Q[(key, best_action)] = (self.N[(key, best_action)] * self.Q[(key, best_action)] + v) / (self.N[(key, best_action)] + 1)
        else:
            self.Q[(key, best_action)] = v
        self.N[(key, best_action)] = self.N.get((key, best_action), 0) + 1
        
        return v

# ==========================================
# 4. INTERFACE (CURSES)
# ==========================================
COLOR_GREY = 1
COLOR_RED = 2
COLOR_CYAN = 3

def print_board(stdscr, board, cursor, ai_last_move=None, msg=""):
    stdscr.clear()
    stdscr.addstr(0, 0, "ROLIT / OTHELLO - ALPHAZERO ARENA")
    stdscr.addstr(1, 0, "Arrows: Move | Enter: Place | Q: Quit")
    
    start_y = 3
    start_x = 2
    
    # Draw Board
    for i in range(8):
        stdscr.addstr(start_y + i, 0, f"{i} ")
        for j in range(8):
            val = board[i, j]
            
            # Determine character
            ch = "."
            if val == 1: ch = "X"
            elif val == 2: ch = "O"
            
            # Determine color
            color = curses.color_pair(COLOR_GREY)
            if val == 1: color = curses.color_pair(COLOR_RED)
            elif val == 2: color = curses.color_pair(COLOR_CYAN)
            
            # Cursor highlight
            if (i, j) == cursor:
                color = color | curses.A_REVERSE
            
            # Last AI move highlight
            if ai_last_move == (i, j):
                ch = "*"
                
            stdscr.addstr(start_y + i, start_x + j*2, ch, color)

    # Score
    p1_score = np.sum(board == 1)
    p2_score = np.sum(board == 2)
    stdscr.addstr(start_y + 9, 0, f"Player 1 (X): {p1_score}")
    stdscr.addstr(start_y + 10, 0, f"Player 2 (O): {p2_score}")
    
    if msg:
        stdscr.addstr(start_y + 12, 0, f">> {msg}", curses.A_BOLD)
    
    stdscr.refresh()

def human_move(stdscr, env, player):
    valid = env.get_legal_moves(player)
    if not valid: return None
    
    cursor = list(valid[0])
    
    while True:
        print_board(stdscr, env.board, tuple(cursor), msg="Your Turn")
        key = stdscr.getch()
        
        if key in [ord('q'), ord('Q')]: 
            return 'QUIT'
        elif key == curses.KEY_UP: 
            cursor[0] = max(0, cursor[0]-1)
        elif key == curses.KEY_DOWN: 
            cursor[0] = min(7, cursor[0]+1)
        elif key == curses.KEY_LEFT: 
            cursor[1] = max(0, cursor[1]-1)
        elif key == curses.KEY_RIGHT: 
            cursor[1] = min(7, cursor[1]+1)
            
        # FIX: Explicitly checking for 10 (Line Feed) and 13 (Carriage Return) 
        # which are the common codes for the main keyboard Enter/Return key.
        elif key in [curses.KEY_ENTER, 10, 13]:
            if tuple(cursor) in valid:
                return tuple(cursor)
            else:
                print_board(
                    stdscr,
                    env.board,
                    tuple(cursor),
                    msg="Illegal move! (press any key)"
                )
                curses.beep()
                stdscr.getch()   # <-- WAIT here

def play_game(model_path="best_alphazero.keras", human_player=1, sims=400):
    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        print("Please ensure you run this script from the same folder as the training script.")
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = RolitEnv(8)
    mcts = MCTS(env, model, simulations=sims)

    def game_loop(stdscr):
        # 1. Initialization and Setup
        curses.curs_set(0) # Hide cursor
        stdscr.keypad(True) # Enable special keys (like arrow keys)
        
        # *** KEY INPUT FIXES ***
        # cbreak(): Pass key presses to the program immediately (no input buffer/line buffering)
        curses.cbreak() 
        # noecho(): Do not echo input characters
        curses.noecho()
        # *** END KEY INPUT FIXES ***

        curses.start_color()
        curses.init_pair(COLOR_GREY, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(COLOR_RED, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(COLOR_CYAN, curses.COLOR_CYAN, curses.COLOR_BLACK)

        board = env.reset()
        player = 1
        ai_last_move = None
        
        # 2. Game Loop
        while not env.done:
            # Check for passed turn
            legal = env.get_legal_moves(player)
            if not legal:
                opp = 2 if player == 1 else 1
                if not env.get_legal_moves(opp): break # Game Over
                print_board(stdscr, board, (-1,-1), ai_last_move, msg=f"Player {player} has no moves! Passing...")
                stdscr.getch()
                player = opp
                continue

            if player == human_player:
                action = human_move(stdscr, env, player)
                if action == 'QUIT': return
                board, _, done = env.step(action, player)
            else:
                print_board(stdscr, board, (-1,-1), ai_last_move, msg=f"AI is thinking ({sims} sims)...")
                
                # Run MCTS
                pi = mcts.run(board, player, env.move_count)
                # Deterministic Play (Argmax) for best performance against Human
                action_idx = np.argmax(pi)
                r, c = divmod(action_idx, 8)
                
                ai_last_move = (r, c)
                board, _, done = env.step((r, c), player)

            player = 2 if player == 1 else 1
            
        # 3. Game Over
        winner = env.winner()
        msg = "Draw!"
        if winner == human_player: msg = "YOU WIN! 🎉"
        elif winner != 0: msg = "AI WINS! 🤖"
        
        # Game over message (wait for Q/q to quit)
        print_board(stdscr, board, (-1,-1), ai_last_move, msg=f"GAME OVER: {msg} (Press 'Q' to quit)")

        while True:
            key = stdscr.getch()
            if key in [ord('q'), ord('Q')]:
                break

        # Re-enable echo and cbreak before exiting curses
        curses.echo()
        curses.nocbreak()

    curses.wrapper(game_loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_alphazero.keras', help='Path to model file')
    parser.add_argument('--player', type=int, default=1, help='Human plays as 1 (X) or 2 (O)')
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations per move (Higher = Stronger but Slower)')
    args = parser.parse_args()
    
    # Ensure the TensorFlow logging is minimal
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    play_game(args.model, args.player, args.sims)