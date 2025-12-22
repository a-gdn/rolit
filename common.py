import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, mixed_precision # type: ignore
import math
from collections import defaultdict
import random
import sys 

# ==========================================
# 0. CONFIGURATION & HELPERS
# ==========================================
BOARD_SIZE = 8
L2_REG = 1e-4
DEFAULT_NUM_PLAYERS = 2
DEFAULT_VARIANT = 'free_rolit' # Options: 'othello', 'rolit', 'free_rolit'

def get_next_player(current_player, num_players):
    """Calculates the ID of the next player."""
    return (current_player % num_players) + 1

def get_user_input(prompt, default, valid_options=None, value_type=str):
    """Helper to ask user for input in the terminal."""
    while True:
        # Format the prompt to show options and default
        options_str = f" [{'/'.join(map(str, valid_options))}]" if valid_options else ""
        full_prompt = f"{prompt}{options_str} (Default: {default}): "
        
        user_in = input(full_prompt).strip()
        
        # If empty, return default
        if not user_in:
            return default
            
        try:
            # Convert type
            val = value_type(user_in)
            
            # Check options
            if valid_options and val not in valid_options:
                print(f"Invalid choice. Please choose from: {valid_options}")
                continue
                
            return val
        except ValueError:
            print(f"Invalid input type. Expected {value_type.__name__}.")

# ==========================================
# 1. GAME LOGIC
# ==========================================
class RolitEnv:
    def __init__(self, board_size=BOARD_SIZE, num_players=DEFAULT_NUM_PLAYERS, variant=DEFAULT_VARIANT):
        self.board_size = board_size
        self.variant = variant.lower()
        
        # Othello strict rule: always 2 players
        if self.variant == 'othello' and num_players != 2:
            print("Warning: Othello variant forces 2 players.")
            num_players = 2

        if not 2 <= num_players <= 4:
            raise ValueError("Implementation supports 2, 3, or 4 players.")
            
        self.num_players = num_players
        # Directions for neighbors
        self.directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        
        # Standard setup for Othello/Rolit/Free Rolit
        # (Diagonal pattern)
        self.board[mid-1, mid-1] = 1 
        self.board[mid, mid] = 2     
        
        if self.num_players == 2:
            self.board[mid-1, mid] = 2 
            self.board[mid, mid-1] = 1 
        elif self.num_players == 3:
            self.board[mid-1, mid] = 3 
            self.board[mid, mid-1] = 1 # Rolit imbalance for 3p start
        elif self.num_players == 4:
            self.board[mid-1, mid] = 3
            self.board[mid, mid-1] = 4

        self.done = False
        self.move_count = 0 
        return self.board.copy()

    def get_legal_moves(self, player):
        """
        Calculates legal moves based on the variant.
        """
        empty_squares = np.argwhere(self.board == 0)
        valid_moves = []
        
        # --- FREE ROLIT: All empty squares are legal ---
        if self.variant == 'free_rolit':
            # Convert numpy array to list of tuples
            return [(r, c) for r, c in empty_squares]
            
        # --- OTHELLO / STANDARD ROLIT: Must capture ---
        for r, c in empty_squares:
            if self._can_flip(r, c, player):
                valid_moves.append((r, c))
                
        return valid_moves

    def _can_flip(self, r, c, player):
        """Checks if placing a piece at (r, c) results in at least one flip."""
        for dr, dc in self.directions:
            if self._find_flip_line(r, c, player, dr, dc, check_only=True):
                return True
        return False

    def _find_flip_line(self, r, c, player, dr, dc, check_only=False):
        """
        Finds pieces to flip along a direction.
        Returns list of (r, c) to flip, or True/False if check_only is True.
        """
        to_flip = []
        temp_r, temp_c = r + dr, c + dc
        
        while 0 <= temp_r < self.board_size and 0 <= temp_c < self.board_size:
            val = self.board[temp_r, temp_c]
            if val == 0:
                break # Hit empty square before closing the line
            if val == player:
                if to_flip:
                    if check_only: return True
                    return to_flip # Line closed, return pieces to flip
                else:
                    return False # Immediate neighbor was own piece
            
            # Opponent piece found
            to_flip.append((temp_r, temp_c))
            temp_r += dr
            temp_c += dc
            
        return False # Did not close the line

    def flip_pieces(self, r, c, player):
        """Attempts to flip pieces in all directions."""
        flipped_any = False
        for dr, dc in self.directions:
            pieces_to_flip = self._find_flip_line(r, c, player, dr, dc, check_only=False)
            if pieces_to_flip:
                flipped_any = True
                for fx, fy in pieces_to_flip:
                    self.board[fx, fy] = player
        return flipped_any

    def step(self, action, player):
        r, c = action
        
        # Place piece
        self.board[r, c] = player
        
        # Attempt to flip (Standard logic applies to all variants IF a line is formed)
        self.flip_pieces(r, c, player)
        
        self.move_count += 1
        
        # Check for game over (no empty squares or no valid moves for any player)
        game_over = not np.any(self.board == 0)
        
        if not game_over:
            # Check if ANY player has a move left
            has_moves = False
            for p in range(1, self.num_players + 1):
                if self.get_legal_moves(p):
                    has_moves = True
                    break
            if not has_moves:
                game_over = True
        
        self.done = game_over
        
        return self.board.copy(), 0, self.done

    def winner(self):
        scores = {p: np.sum(self.board == p) for p in range(1, self.num_players + 1)}
        if sum(scores.values()) == 0: return 0
        
        max_score = -1
        winner = 0
        for player_id, score in scores.items():
            if score > max_score:
                max_score = score
                winner = player_id
            elif score == max_score:
                return 0 # Draw
        return winner

# ==========================================
# 2. NEURAL NETWORK HELPERS
# ==========================================
def preprocess_state(board, player, move_count, num_players):
    """
    State representation: N player-specific planes + 1 game progress plane.
    """
    total_planes = num_players + 1
    state = np.zeros((BOARD_SIZE, BOARD_SIZE, total_planes), dtype=np.float32)
    
    # Piece Planes (P0 = Current Player, P1 = Next Player...)
    player_list = [((player + i - 1) % num_players) + 1 for i in range(num_players)]
    
    for i, p_id in enumerate(player_list):
        state[:, :, i] = (board == p_id).astype(np.float32)

    # Time/Turn Plane
    max_cells = BOARD_SIZE * BOARD_SIZE
    state[:, :, num_players] = move_count / max_cells
    
    return state

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization(dtype='float32')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization(dtype='float32')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_planes, residual_blocks, learning_rate, board_size=BOARD_SIZE):
    inputs = layers.Input(shape=(board_size, board_size, input_planes)) 
    
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization(dtype='float32')(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(residual_blocks):
        x = residual_block(x, 64)
        
    # Policy Head
    ph = layers.Conv2D(2, (1,1), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    ph = layers.BatchNormalization(dtype='float32')(ph)
    ph = layers.Activation('relu')(ph)
    ph = layers.Flatten()(ph)
    ph = layers.Dense(board_size * board_size, activation='softmax', name='policy', dtype='float32')(ph)
    
    # Value Head
    vh = layers.Conv2D(1, (1,1), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    vh = layers.BatchNormalization(dtype='float32')(vh)
    vh = layers.Activation('relu')(vh)
    vh = layers.Flatten()(vh)
    vh = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(vh)
    vh = layers.Dense(1, activation='tanh', name='value', dtype='float32')(vh)
    
    model = models.Model(inputs=inputs, outputs=[ph, vh])

    opt = optimizers.Adam(learning_rate)
    try:
        if mixed_precision.global_policy().name == 'mixed_float16':
            opt = mixed_precision.LossScaleOptimizer(opt)
    except Exception:
        pass

    model.compile(optimizer=opt, 
                  loss={'policy': 'categorical_crossentropy', 'value': 'mse'})
    return model

# ==========================================
# 3. MCTS LOGIC
# ==========================================
class MCTS:
    def __init__(self, env, model, simulations=400, cpuct=2.0, num_players=DEFAULT_NUM_PLAYERS):
        self.env = env
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        self.num_players = num_players
        self.Q = defaultdict(float) 
        self.N = defaultdict(float) 
        self.P = {}

        # Compiled inference function for faster repeated evaluations
        try:
            # Use tf.function to compile the model call with training=False
            self._infer = tf.function(lambda x: self.model(x, training=False))
        except Exception:
            # Fallback to direct call if compilation is not possible
            self._infer = lambda x: self.model(x, training=False)

    def run(self, board, player, move_count, is_self_play=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        # Clear tree if starting new self-play game to prevent memory bloat over long runs
        if is_self_play and move_count == 0:
             self.Q.clear(); self.N.clear(); self.P.clear()

        for _ in range(self.simulations):
            self._simulate(board.copy(), player, move_count)
            
        pi = np.zeros(self.env.board_size**2)
        root = (board.tobytes(), player)
        legal = self.env.get_legal_moves(player)
        
        for r, c in legal:
            action = r * self.env.board_size + c
            pi[action] = self.N.get((root, action), 0)
            
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        
        if is_self_play and legal:
            pi_legal = [pi[r * self.env.board_size + c] for r, c in legal]
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
            
            new_pi_legal = (1 - dirichlet_epsilon) * np.array(pi_legal) + dirichlet_epsilon * noise
            
            new_pi = np.zeros_like(pi)
            for i, (r, c) in enumerate(legal):
                new_pi[r * self.env.board_size + c] = new_pi_legal[i]
            
            # Re-normalize
            sum_new = np.sum(new_pi)
            if sum_new > 0:
                pi = new_pi / sum_new

        return pi

    def _simulate(self, board, player, move_count):
        key = (board.tobytes(), player)
        
        # 1. CHECK TERMINAL
        # Use a temporary env to check game state. 
        # Crucially, pass the same variant and player count
        temp_env = RolitEnv(self.env.board_size, self.num_players, self.env.variant)
        temp_env.board = board
        legal = temp_env.get_legal_moves(player) 
        
        if not legal:
            # Check for global game over or just a forced pass
            is_game_over = not np.any(board == 0)
            
            if not is_game_over:
                # Check if ANY player can make a move
                has_any_move = False
                for p in range(1, self.num_players + 1):
                    if temp_env.get_legal_moves(p):
                        has_any_move = True
                        break
                if not has_any_move:
                    is_game_over = True

            if is_game_over:
                w = temp_env.winner()
                if w == 0: return 0 
                return 1 if w == player else -1
            
            # Pass turn (Recursion)
            next_player = get_next_player(player, self.num_players)
            # Invert value because opponent's gain is our loss
            return -self._simulate(board, next_player, move_count)

        # 2. EXPANSION
        if key not in self.P:
            state_input = preprocess_state(board, player, move_count, self.num_players)[np.newaxis, ...]
            predictions = self._infer(state_input)
            policy = predictions[0].numpy()[0]
            value = float(predictions[1].numpy()[0][0])

            # Mask illegal moves
            mask = np.zeros_like(policy)
            for r, c in legal:
                mask[r * self.env.board_size + c] = 1
            policy = policy * mask
            
            pol_sum = np.sum(policy)
            if pol_sum > 0:
                policy /= pol_sum
            else:
                # If network predicts 0 for all legal moves, treat uniformly
                policy = mask / np.sum(mask)
                
            self.P[key] = policy
            return value

        # 3. SELECTION
        total_N = sum(self.N.get((key, r * self.env.board_size + c), 0) for r, c in legal)
        sqrt_total = math.sqrt(total_N + 1e-8)
        
        best_u = -float('inf')
        best_action = None
        
        for r, c in legal:
            a = r * self.env.board_size + c
            q = self.Q.get((key, a), 0)
            n = self.N.get((key, a), 0)
            p = self.P[key][a]
            
            u = q + self.cpuct * p * sqrt_total / (1 + n)
            
            if u > best_u:
                best_u = u
                best_action = a
                
        # 4. STEP
        r, c = divmod(best_action, self.env.board_size)
        
        next_env = RolitEnv(self.env.board_size, self.num_players, self.env.variant)
        next_env.board = board.copy()
        next_env.move_count = move_count 
        
        next_board, _, _ = next_env.step((r, c), player)
        next_player = get_next_player(player, self.num_players)
        
        val = self._simulate(next_board, next_player, next_env.move_count) 
        
        # 5. BACKPROP
        v = -val 
        
        old_q = self.Q.get((key, best_action), 0)
        old_n = self.N.get((key, best_action), 0)
        
        self.Q[(key, best_action)] = (old_n * old_q + v) / (old_n + 1)
        self.N[(key, best_action)] += 1
        
        return v