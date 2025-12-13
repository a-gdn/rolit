import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import math
from collections import defaultdict
import random

# ==========================================
# 0. CONFIGURATION & HELPERS
# ==========================================
BOARD_SIZE = 8
L2_REG = 1e-4
DEFAULT_NUM_PLAYERS = 2

def get_next_player(current_player, num_players):
    """Calculates the ID of the next player."""
    return (current_player % num_players) + 1

# ==========================================
# 1. GAME LOGIC (ROLIT)
# ==========================================
class RolitEnv:
    def __init__(self, board_size=BOARD_SIZE, num_players=DEFAULT_NUM_PLAYERS):
        self.board_size = board_size
        if not 2 <= num_players <= 4:
            raise ValueError("Rolit implementation supports 2, 3, or 4 players.")
        self.num_players = num_players
        # Directions for neighbors
        self.directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        mid = self.board_size // 2
        
        # Initial 4 central pieces placed in a diagonal pattern
        self.board[mid-1, mid-1] = 1 # Player 1 (Top-Left)
        self.board[mid, mid] = 2     # Player 2 (Bottom-Right)
        self.board[mid-1, mid] = 3 if self.num_players >= 3 else 1 # Player 3 (Top-Right)
        self.board[mid, mid-1] = 4 if self.num_players == 4 else 2 # Player 4 (Bottom-Left)
        
        # Adjusting the starting pieces for 2/3 player games to ensure 4 pieces total
        if self.num_players == 2:
            self.board[mid-1, mid] = 2
            self.board[mid, mid-1] = 1
        elif self.num_players == 3:
            # Set up P1, P2, P3, P1 in the four central squares
            self.board[mid, mid-1] = 1 # Override P4 spot with P1
        
        self.done = False
        self.move_count = 0 
        return self.board.copy()

    def get_legal_moves(self, player):
        """
        Rolit Rule: Unlike Othello, any free square is a legal move.
        However, a move *must* result in at least one flip.
        If no squares result in a flip, the player must pass, but the game is not over.
        """
        empty_squares = np.argwhere(self.board == 0)
        
        valid_moves = []
        for r, c in empty_squares:
            # Check if this square can flip any opponent pieces
            if self._can_flip(r, c, player):
                valid_moves.append((r, c))
                
        # If there are no valid *flipping* moves, the player must pass, but the game is not over yet.
        # This function should only return moves that result in a flip for MCTS/game logic.
        return valid_moves if valid_moves else []

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
                if check_only and to_flip:
                    return True
                return to_flip # Line closed, return pieces to flip
            
            # Opponent piece found
            to_flip.append((temp_r, temp_c))
            temp_r += dr
            temp_c += dc
            
        return False # Did not close the line

    def flip_pieces(self, r, c, player):
        for dr, dc in self.directions:
            pieces_to_flip = self._find_flip_line(r, c, player, dr, dc, check_only=False)
            if pieces_to_flip:
                for fx, fy in pieces_to_flip:
                    self.board[fx, fy] = player

    def step(self, action, player):
        r, c = action
        self.board[r, c] = player
        self.flip_pieces(r, c, player)
        self.move_count += 1
        
        # Check for game over (no empty squares or no valid moves for any player)
        game_over = not np.any(self.board == 0)
        
        # Also check if any player has a valid move. If the current player has no moves, 
        # the next player gets a chance, and so on.
        if not game_over:
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
        
        # Check if any pieces were placed at all
        if sum(scores.values()) == 0: return 0 # No pieces, shouldn't happen
        
        # Find the player with the maximum score
        max_score = -1
        winner = 0
        
        for player_id, score in scores.items():
            if score > max_score:
                max_score = score
                winner = player_id
            elif score == max_score:
                # Tie: return 0 (Draw)
                return 0
        
        return winner

# ==========================================
# 2. NEURAL NETWORK HELPERS
# ==========================================
def preprocess_state(board, player, move_count, num_players=DEFAULT_NUM_PLAYERS):
    """
    State representation:
    Plane 0: My pieces (P_current)
    Plane 1: Opponent 1 pieces (P_current + 1)
    ...
    Plane N-1: Opponent N-1 pieces (P_current - 1)
    Plane N: Game Progress/Turn count (Scaled from 0.0 to 1.0)
    
    Total planes = num_players + 1
    """
    total_planes = num_players + 1
    state = np.zeros((BOARD_SIZE, BOARD_SIZE, total_planes), dtype=np.float32)
    
    # Piece Planes (P0 to P_N-1)
    player_list = [((player + i - 1) % num_players) + 1 for i in range(num_players)]
    
    for i, p_id in enumerate(player_list):
        state[:, :, i] = (board == p_id).astype(np.float32)

    # Time/Turn Plane (P_N)
    max_cells = BOARD_SIZE * BOARD_SIZE
    state[:, :, num_players] = move_count / max_cells
    
    return state

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_planes, residual_blocks, learning_rate, board_size=BOARD_SIZE):
    inputs = layers.Input(shape=(board_size, board_size, input_planes)) 
    
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(residual_blocks):
        x = residual_block(x, 64)
        
    # Policy Head
    ph = layers.Conv2D(2, (1,1), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    ph = layers.BatchNormalization()(ph)
    ph = layers.Activation('relu')(ph)
    ph = layers.Flatten()(ph)
    ph = layers.Dense(board_size * board_size, activation='softmax', name='policy')(ph)
    
    # Value Head
    vh = layers.Conv2D(1, (1,1), padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    vh = layers.BatchNormalization()(vh)
    vh = layers.Activation('relu')(vh)
    vh = layers.Flatten()(vh)
    vh = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(vh)
    vh = layers.Dense(1, activation='tanh', name='value')(vh)
    
    model = models.Model(inputs=inputs, outputs=[ph, vh])
    model.compile(optimizer=optimizers.Adam(learning_rate), 
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
        # Using dicts for Q, N, P to store tree data
        self.Q = defaultdict(float) # Q(s, a): average value of action a from state s
        self.N = defaultdict(float) # N(s, a): visit count of action a from state s
        self.P = {}                 # P(s): prior policy (from model) for state s

    def run(self, board, player, move_count, is_self_play=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        key = (board.tobytes(), player)
        
        # Clear tree only at the start of a new game for self-play
        if is_self_play and move_count == 0:
             self.Q.clear(); self.N.clear(); self.P.clear()

        # Run simulations
        for _ in range(self.simulations):
            self._simulate(board.copy(), player, move_count)
            
        pi = np.zeros(self.env.board_size**2)
        root = (board.tobytes(), player)
        legal = self.env.get_legal_moves(player)
        
        # Calculate policy from visit counts
        for r, c in legal:
            action = r * self.env.board_size + c
            pi[action] = self.N.get((root, action), 0)
            
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        
        # Apply Dirichlet Noise ONLY during self-play for exploration
        if is_self_play and legal:
            pi_legal = [pi[r * self.env.board_size + c] for r, c in legal]
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
            
            new_pi_legal = (1 - dirichlet_epsilon) * np.array(pi_legal) + dirichlet_epsilon * noise
            
            # Map back to the full 64-length vector
            new_pi = np.zeros_like(pi)
            for i, (r, c) in enumerate(legal):
                new_pi[r * self.env.board_size + c] = new_pi_legal[i]
            
            pi = new_pi / np.sum(new_pi)

        return pi

    def _simulate(self, board, player, move_count):
        key = (board.tobytes(), player)
        
        # 1. CHECK TERMINAL
        temp_env = RolitEnv(self.env.board_size, self.num_players)
        temp_env.board = board
        legal = temp_env.get_legal_moves(player)
        
        if not legal:
            # Current player must pass. Check if the game is truly over.
            next_player = get_next_player(player, self.num_players)
            
            # If next player also has no moves, check the one after that, up to all players.
            # This handles the Rolit pass-rule until all players pass or the board fills.
            
            # Check for global game over (no moves for ANY player, or board full)
            if temp_env.done:
                w = temp_env.winner()
                if w == 0: return 0 # Draw
                return 1 if w == player else -1 # Win/Loss relative to current 'player'

            # Game is NOT over, pass the turn and recurse immediately.
            # The result must be negated because the board state didn't change, 
            # but the perspective shifted to the new player.
            return -self._simulate(board, next_player, move_count)

        # 2. EXPANSION (Leaf Node)
        if key not in self.P:
            # Predict policy and value from the neural network
            state_input = preprocess_state(board, player, move_count, self.num_players)[np.newaxis, ...]
            policy, value = self.model.predict(state_input, verbose=0)
            policy = policy[0]
            value = value[0][0]
            
            # Mask illegal moves (Crucial for correct policy representation)
            mask = np.zeros_like(policy)
            for r, c in legal:
                mask[r * self.env.board_size + c] = 1
            policy = policy * mask
            
            pol_sum = np.sum(policy)
            if pol_sum > 0:
                policy /= pol_sum
            else:
                # If NN gives zero probability to all legal moves, assign uniform probability
                policy = mask / np.sum(mask)
                
            self.P[key] = policy
            return value

        # 3. SELECTION (PUCT)
        total_N = sum(self.N.get((key, r * self.env.board_size + c), 0) for r, c in legal)
        sqrt_total = math.sqrt(total_N + 1e-8)
        
        best_u = -float('inf')
        best_action = None
        
        for r, c in legal:
            a = r * self.env.board_size + c
            q = self.Q.get((key, a), 0)
            n = self.N.get((key, a), 0)
            p = self.P[key][a]
            
            # PUCT formula: Q + C_puct * P * sqrt(N_total) / (1 + N_action)
            u = q + self.cpuct * p * sqrt_total / (1 + n)
            
            if u > best_u:
                best_u = u
                best_action = a
                
        # 4. STEP AND RECURSE
        r, c = divmod(best_action, self.env.board_size)
        
        # Create a new environment for the next state
        next_env = RolitEnv(self.env.board_size, self.num_players)
        next_env.board = board.copy()
        next_env.move_count = move_count 
        
        next_board, _, _ = next_env.step((r, c), player)
        next_player = get_next_player(player, self.num_players)
        
        # Recurse: Get the value (W) of the resulting state from the *next player's* perspective
        val = self._simulate(next_board, next_player, next_env.move_count) 
        
        # 5. BACKPROPAGATION
        # The value (V) of the current state for the current player is the negative 
        # of the resulting state's value (W) from the next player's perspective. V = -W.
        v = -val 
        
        # Update Q and N
        old_q = self.Q.get((key, best_action), 0)
        old_n = self.N.get((key, best_action), 0)
        
        self.Q[(key, best_action)] = (old_n * old_q + v) / (old_n + 1)
        self.N[(key, best_action)] += 1
        
        return v