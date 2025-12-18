import numpy as np
import tensorflow as tf
import curses
import argparse
import os
import time

# Import common game logic and MCTS implementation
from common import RolitEnv, MCTS, get_next_player, DEFAULT_NUM_PLAYERS, DEFAULT_VARIANT, BOARD_SIZE

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
    
    COLOR_MAP = {1: COLOR_P1, 2: COLOR_P2, 3: COLOR_P3, 4: COLOR_P4}
    char_map = {1: 'X', 2: 'O', 3: 'A', 4: 'B'}
    player_chars = {p: char_map[p] for p in range(1, env.num_players + 1)}
    
    # Title
    stdscr.addstr(0, 0, f"{env.variant.upper()} - {env.num_players} PLAYER ARENA")
    stdscr.addstr(1, 0, "Arrows: Move | Enter: Place | Q: Quit")
    
    start_y = 3
    start_x = 2
    
    # Draw Board
    for i in range(BOARD_SIZE):
        stdscr.addstr(start_y + i, 0, f"{i} ")
        for j in range(BOARD_SIZE):
            val = env.board[i, j]
            ch = "."
            color = curses.color_pair(COLOR_GREY)
            
            if val in player_chars:
                ch = player_chars[val]
                color = curses.color_pair(COLOR_MAP.get(val, COLOR_GREY))
            
            if (i, j) == cursor:
                color = color | curses.A_REVERSE
            
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
    
    cursor = list(valid[0]) if valid else [0, 0]
    
    # If using Free Rolit, set cursor to any empty spot if valid list is huge
    if env.variant == 'free_rolit' and not env.board[0,0]:
         cursor = [0,0]

    while True:
        print_board(stdscr, env, player, tuple(cursor), msg=f"Player {player} ({env.variant}): Your Turn")
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
                if env.board[cursor[0], cursor[1]] != 0:
                    msg = "Square already occupied!"
                else:
                    msg = "Illegal move! Rules require a flip here."
                    
                print_board(stdscr, env, player, tuple(cursor), msg=msg)
                curses.beep()
                stdscr.getch()

def play_game(model_path, human_player, sims, num_players, variant):
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = RolitEnv(BOARD_SIZE, num_players, variant)
    mcts = MCTS(env, model, simulations=sims, num_players=num_players)

    def game_loop(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.cbreak() 
        curses.noecho()
        curses.start_color()
        curses.init_pair(COLOR_GREY, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(COLOR_P1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(COLOR_P2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        if num_players >= 3 and curses.COLORS >= 8:
            curses.init_pair(COLOR_P3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        if num_players == 4 and curses.COLORS >= 8:
            curses.init_pair(COLOR_P4, curses.COLOR_GREEN, curses.COLOR_BLACK)

        env.reset()
        player = 1
        ai_last_move = None
        
        while not env.done:
            legal = env.get_legal_moves(player)
            
            if not legal:
                if env.done: break 
                print_board(stdscr, env, player, (-1,-1), ai_last_move, 
                            msg=f"Player {player} has no legal moves (must pass)...")
                stdscr.getch()
                player = get_next_player(player, env.num_players)
                continue

            if player == human_player:
                action = human_move(stdscr, env, player)
                if action == 'QUIT': return
                if action == 'PASS':
                    player = get_next_player(player, env.num_players)
                    continue
                env.step(action, player)
            else:
                print_board(stdscr, env, player, (-1,-1), ai_last_move, msg=f"AI is thinking ({sims} sims)...")
                pi = mcts.run(env.board, player, env.move_count)
                action_idx = np.argmax(pi)
                r, c = divmod(action_idx, BOARD_SIZE)
                ai_last_move = (r, c)
                env.step((r, c), player)

            player = get_next_player(player, env.num_players)
            
        winner = env.winner()
        if winner == 0: msg = "Draw!"
        elif winner == human_player: msg = "YOU WIN! 🎉"
        else: msg = f"AI P{winner} WINS! 🤖"
        
        print_board(stdscr, env, player, (-1,-1), ai_last_move, msg=f"GAME OVER: {msg} (Press 'Q' to quit)")
        while True:
            if stdscr.getch() in [ord('q'), ord('Q')]: break

        curses.echo()
        curses.nocbreak()

    curses.wrapper(game_loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default=DEFAULT_VARIANT, choices=['othello', 'rolit', 'free_rolit'], help='Game variant')
    parser.add_argument('--player', type=int, default=1, choices=[1, 2, 3, 4], help='Human player ID')
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--num_players', type=int, default=DEFAULT_NUM_PLAYERS, choices=[2, 3, 4])
    args = parser.parse_args()
    
    # Enforce Othello constraint
    if args.variant == 'othello' and args.num_players != 2:
        print("Forcing 2 players for Othello.")
        args.num_players = 2

    # Path Construction
    model_dir = "models"
    model_filename = f"best_model_{args.variant}_{args.num_players}p.keras"
    model_path = os.path.join(model_dir, model_filename)
    
    # Comprehensive Existence Check
    if not os.path.isdir(model_dir):
        print(f"Error: The directory '{model_dir}' does not exist.")
        exit(1)
    
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Make sure you have trained a {args.num_players}-player model for {args.variant}.")
        exit(1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    play_game(model_path, args.player, args.sims, args.num_players, args.variant)