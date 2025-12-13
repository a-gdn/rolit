import numpy as np
import tensorflow as tf
import curses
import argparse
import os
import math
import time

# Import common game logic and MCTS implementation
from common import RolitEnv, MCTS, preprocess_state, get_next_player, DEFAULT_NUM_PLAYERS, BOARD_SIZE

# ==========================================
# 1. INTERFACE (CURSES)
# ==========================================
COLOR_GREY = 1
COLOR_P1 = 2 # Red
COLOR_P2 = 3 # Cyan
COLOR_P3 = 4 # Yellow
COLOR_P4 = 5 # Green

def print_board(stdscr, env, player, cursor, ai_last_move=None, msg=""):
    stdscr.clear()
    
    # Player ID to color mapping
    COLOR_MAP = {1: COLOR_P1, 2: COLOR_P2, 3: COLOR_P3, 4: COLOR_P4}
    
    # Get Player Character/Name
    char_map = {1: 'X', 2: 'O', 3: 'A', 4: 'B'}
    player_chars = {p: char_map[p] for p in range(1, env.num_players + 1)}
    
    # Title
    stdscr.addstr(0, 0, f"ROLIT - {env.num_players} PLAYER ARENA")
    stdscr.addstr(1, 0, "Arrows: Move | Enter: Place | Q: Quit")
    
    start_y = 3
    start_x = 2
    
    # Draw Board
    for i in range(BOARD_SIZE):
        stdscr.addstr(start_y + i, 0, f"{i} ")
        for j in range(BOARD_SIZE):
            val = env.board[i, j]
            
            # Determine character and color
            ch = "."
            color = curses.color_pair(COLOR_GREY)
            
            if val in player_chars:
                ch = player_chars[val]
                color = curses.color_pair(COLOR_MAP.get(val, COLOR_GREY))
            
            # Cursor highlight
            if (i, j) == cursor:
                color = color | curses.A_REVERSE
            
            # Last AI move highlight
            if ai_last_move == (i, j):
                ch = "*"
                
            stdscr.addstr(start_y + i, start_x + j*2, ch, color)

    # Score
    row_offset = 9
    scores = {p: np.sum(env.board == p) for p in range(1, env.num_players + 1)}
    for p in range(1, env.num_players + 1):
        color = curses.color_pair(COLOR_MAP.get(p, COLOR_GREY))
        stdscr.addstr(start_y + row_offset, 0, f"Player {p} ({player_chars[p]}): {scores[p]}", color)
        row_offset += 1
    
    if msg:
        stdscr.addstr(start_y + row_offset + 1, 0, f">> {msg}", curses.A_BOLD)
    
    stdscr.refresh()

def human_move(stdscr, env, player):
    valid = env.get_legal_moves(player)
    
    if not valid: 
        return 'PASS'
    
    # Find a legal starting cursor position, or default to a corner
    cursor = list(valid[0]) if valid else [0, 0] 
    
    while True:
        print_board(stdscr, env, player, tuple(cursor), msg=f"Player {player} ({env.num_players}P): Your Turn")
        key = stdscr.getch()
        
        if key in [ord('q'), ord('Q')]: 
            return 'QUIT'
        elif key == curses.KEY_UP: 
            cursor[0] = max(0, cursor[0]-1)
        elif key == curses.KEY_DOWN: 
            cursor[0] = min(BOARD_SIZE-1, cursor[0]+1)
        elif key == curses.KEY_LEFT: 
            cursor[1] = max(0, cursor[1]-1)
        elif key == curses.KEY_RIGHT: 
            cursor[1] = min(BOARD_SIZE-1, cursor[1]+1)
            
        elif key in [curses.KEY_ENTER, 10, 13]:
            if tuple(cursor) in valid:
                return tuple(cursor)
            else:
                # Rolit rule: all empty squares are visually selectable, 
                # but must result in a flip (must be in 'valid' list)
                if env.board[cursor[0], cursor[1]] != 0:
                    msg = "Square already occupied!"
                else:
                    msg = "Illegal move! Must flip opponent pieces. (press any key)"
                    
                print_board(
                    stdscr,
                    env,
                    player,
                    tuple(cursor),
                    msg=msg
                )
                curses.beep()
                stdscr.getch()

def play_game(model_path, human_player, sims, num_players):
    # Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = RolitEnv(BOARD_SIZE, num_players)
    mcts = MCTS(env, model, simulations=sims, num_players=num_players)

    def game_loop(stdscr):
        # 1. Initialization and Setup
        curses.curs_set(0) # Hide cursor
        stdscr.keypad(True) # Enable special keys
        curses.cbreak() 
        curses.noecho()

        curses.start_color()
        curses.init_pair(COLOR_GREY, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(COLOR_P1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(COLOR_P2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        # Initialize colors for 3 and 4 players (if supported by terminal)
        if num_players >= 3 and curses.COLORS >= 8:
            curses.init_pair(COLOR_P3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        if num_players == 4 and curses.COLORS >= 8:
            curses.init_pair(COLOR_P4, curses.COLOR_GREEN, curses.COLOR_BLACK)

        env.reset()
        player = 1
        ai_last_move = None
        
        # 2. Game Loop
        while not env.done:
            legal = env.get_legal_moves(player)
            
            if not legal:
                # Rolit Rule: Player must pass if no legal moves exist.
                
                # Check for global game over
                if env.done: break 
                
                print_board(stdscr, env, player, (-1,-1), ai_last_move, 
                            msg=f"Player {player} has no legal moves (must pass)...")
                stdscr.getch()
                
                player = get_next_player(player, env.num_players)
                continue

            if player == human_player:
                action = human_move(stdscr, env, player)
                
                if action == 'QUIT': return
                if action == 'PASS': # Should not happen if legal list is not empty, but for safety
                    player = get_next_player(player, env.num_players)
                    continue

                env.step(action, player)
            else:
                print_board(stdscr, env, player, (-1,-1), ai_last_move, msg=f"AI is thinking ({sims} sims) for P{player}...")
                
                # Run MCTS
                pi = mcts.run(env.board, player, env.move_count)
                # Deterministic Play (Argmax)
                action_idx = np.argmax(pi)
                r, c = divmod(action_idx, BOARD_SIZE)
                action = (r, c)
                
                ai_last_move = action
                env.step(action, player)

            player = get_next_player(player, env.num_players)
            
        # 3. Game Over
        winner = env.winner()
        
        if winner == 0: msg = "Draw!"
        elif winner == human_player: msg = "YOU WIN! 🎉"
        else: msg = f"AI P{winner} WINS! 🤖"
        
        # Game over message (wait for Q/q to quit)
        print_board(stdscr, env, player, (-1,-1), ai_last_move, msg=f"GAME OVER: {msg} (Press 'Q' to quit)")

        while True:
            key = stdscr.getch()
            if key in [ord('q'), ord('Q')]:
                break

        # Re-enable echo and cbreak before exiting curses
        curses.echo()
        curses.nocbreak()

    curses.wrapper(game_loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a Rolit game against the trained agent.')
    parser.add_argument('--model', default='best_rolit_model.keras', help='Path to model file')
    parser.add_argument('--player', type=int, default=1, choices=[1, 2, 3, 4], help='Human plays as P1, P2, etc.')
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations per move (Higher = Stronger but Slower)')
    parser.add_argument('--num_players', type=int, default=DEFAULT_NUM_PLAYERS, choices=[2, 3, 4], help='Number of players in the game.')
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    play_game(args.model, args.player, args.sims, args.num_players)