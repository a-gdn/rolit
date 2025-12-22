import random
import numpy as np
import tensorflow as tf
from collections import deque
import os
import sys # Added sys
import pickle
import subprocess
import atexit
import argparse

from common import (
    RolitEnv, MCTS, preprocess_state, build_model, get_next_player, get_user_input,
    BOARD_SIZE, DEFAULT_NUM_PLAYERS, DEFAULT_VARIANT
)

# ==========================================
# 0. SYSTEM SETUP
# ==========================================
try:
    caffeinate_proc = subprocess.Popen(['caffeinate', '-dimsu'])
    atexit.register(lambda: caffeinate_proc.terminate())
except Exception:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU / Metal (Apple) setup: enable memory growth and optional mixed precision
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("Detected GPU(s):", gpus)
    except Exception as e:
        print("GPU setup error:", e)
else:
    print("No GPU detected; running on CPU.")

# Enable mixed precision on capable hardware for better performance
from tensorflow.keras import mixed_precision # type: ignore
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision policy:", mixed_precision.global_policy().name)

# ==========================================
# 1. CONFIGURATION
# ==========================================
LEARNING_RATE = 0.001
ITERATIONS = 100         
EPISODES_PER_ITER = 10  
MCTS_SIMS = 400          
BATCH_SIZE = 256
EPOCHS = 3
BUFFER_SIZE = 100000 
RESIDUAL_BLOCKS = 5  
ARENA_FREQ = 3

DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25 

# ==========================================
# 2. REPLAY BUFFER & SELF-PLAY
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, s, p, z):
        self.buffer.append((s, p, z))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s, p, z = zip(*batch)
        return np.array(s), np.array(p), np.array(z)
    
    def __len__(self):
        return len(self.buffer)

def self_play(env, model, sims):
    env.reset()
    board = env.board.copy()
    player = 1
    mcts = MCTS(env, model, simulations=sims, num_players=env.num_players) 
    trajectory = []
    
    while True:
        legal = env.get_legal_moves(player)
        
        if not legal:
            if env.done: break 
            player = get_next_player(player, env.num_players)
            continue

        pi = mcts.run(board, player, env.move_count, is_self_play=True,
                      dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON)
        trajectory.append((board.copy(), pi, player, env.move_count))
        
        if env.move_count < 12: 
            action = np.random.choice(len(pi), p=pi)
        else:
            action = np.argmax(pi)
            
        r, c = divmod(action, env.board_size)
        env.step((r, c), player)
        board = env.board.copy()
        
        if env.done: break
        player = get_next_player(player, env.num_players)
        
    winner = env.winner()
    data = []
    
    for s, pi, turn, mc in trajectory:
        if winner == 0: z = 0
        else: z = 1 if winner == turn else -1
            
        state = preprocess_state(s, turn, mc, env.num_players) 
        
        if env.board_size == 8:
            pi_board = pi.reshape(BOARD_SIZE, BOARD_SIZE)
            for k in range(4):
                rot_state = np.rot90(state, k)
                rot_pi = np.rot90(pi_board, k)
                data.append((rot_state, rot_pi.flatten(), z))
                data.append((np.fliplr(rot_state), np.fliplr(rot_pi).flatten(), z))
        else:
            data.append((state, pi, z)) 
    return data

# ==========================================
# 3. ARENA EVALUATION
# ==========================================
def evaluate_vs_best(env, current_model, best_model_path, sims, games):
    if not os.path.exists(best_model_path):
        return 1.0

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    wins, draws = 0, 0
    
    for i in range(games):
        print(f"Arena: Game {i + 1}/{games}...", end='\r')
        start_player_current = (i % env.num_players) + 1
        
        models_map = {}
        for p in range(1, env.num_players + 1):
            models_map[p] = current_model if p == start_player_current else best_model
        
        env.reset()
        board = env.board.copy()
        player = 1
        agents = {p: MCTS(env, models_map[p], simulations=sims, num_players=env.num_players) for p in range(1, env.num_players+1)}
        
        while not env.done:
            legal = env.get_legal_moves(player)
            if not legal:
                if env.done: break
                player = get_next_player(player, env.num_players)
                continue
            
            pi = agents[player].run(board, player, env.move_count, is_self_play=False)
            action = np.argmax(pi)
            r, c = divmod(action, env.board_size)
            env.step((r, c), player)
            board = env.board.copy()
            player = get_next_player(player, env.num_players)

        winner = env.winner()
        if winner == 0: draws += 1
        elif winner == start_player_current: wins += 1
    
    print("")
    return (wins + 0.5 * draws) / games

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_players', type=int, choices=[2, 3, 4])
    parser.add_argument('--variant', choices=['othello', 'rolit', 'free_rolit'])
    args = parser.parse_args()
    
    # --- INTERACTIVE WIZARD ---
    # If arguments are missing, ask the user interactively
    if args.variant is None:
        args.variant = get_user_input("Select Variant", DEFAULT_VARIANT, ['othello', 'rolit', 'free_rolit'])
    
    if args.num_players is None:
        if args.variant == 'othello':
            print("Othello automatically selected for 2 players.")
            args.num_players = 2
        else:
            args.num_players = get_user_input("Select Number of Players", DEFAULT_NUM_PLAYERS, [2, 3, 4], int)
    
    if args.variant == 'othello' and args.num_players != 2:
        print("Warning: Othello forces 2 players. Resetting to 2.")
        args.num_players = 2
    
    # Unique identifiers for file paths
    ID_STR = f"{args.variant}_{args.num_players}p"
    
    # --- UPDATED PATHS ---
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(MODELS_DIR, f"model_{ID_STR}.keras")      
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, f"best_{ID_STR}.keras") 
    BUFFER_PATH = os.path.join(MODELS_DIR, f"buffer_{ID_STR}.pkl")     
    LOG_DIR = f"logs/{ID_STR}"
    
    env = RolitEnv(BOARD_SIZE, args.num_players, args.variant)
    
    # 1. BUFFER
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, "rb") as f:
            buffer = pickle.load(f)
        print(f"Loaded buffer for {ID_STR}: {len(buffer)} samples.")
    else:
        print(f"No buffer found at {BUFFER_PATH}. Starting with empty buffer.")
        buffer = ReplayBuffer(BUFFER_SIZE)

    # 2. MODEL
    INPUT_PLANES = args.num_players + 1
    
    # Priority 1: Resume from current checkpoint
    if os.path.exists(MODEL_PATH):
        print(f"Loaded existing checkpoint: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Priority 2: Resume from Best Model (Hot Start)
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"No checkpoint found, but found best model: {BEST_MODEL_PATH}")
        print("Loading best model to resume training...")
        model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)

    # Priority 3: Start from scratch
    else:
        print(f"No models found. Creating new model: {MODEL_PATH}")
        model = build_model(INPUT_PLANES, RESIDUAL_BLOCKS, LEARNING_RATE)

    # Configure optimizer to handle mixed precision if enabled
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    try:
        from tensorflow.keras import mixed_precision as _mp
        if _mp.global_policy().name == 'mixed_float16':
            opt = _mp.LossScaleOptimizer(opt)
    except Exception:
        pass

    model.compile(optimizer=opt, 
                  loss={'policy': 'categorical_crossentropy', 'value': 'mse'})

    if not os.path.exists(BEST_MODEL_PATH):
        model.save(BEST_MODEL_PATH)

    print(f"\nStarting Training: {ITERATIONS} Iterations | Variant: {args.variant} | Players: {args.num_players}")

    for iteration in range(ITERATIONS):
        print(f"\n--- Iteration {iteration+1}/{ITERATIONS} ---")
        
        # Self Play
        new_samples = 0
        for i in range(EPISODES_PER_ITER):
            data = self_play(env, model, MCTS_SIMS) 
            for sample in data:
                buffer.add(*sample)
            new_samples += len(data)
            print(f"Self-Play {i+1}/{EPISODES_PER_ITER}: Buffer={len(buffer)}", end='\r')
        
        # Train
        if len(buffer) > BATCH_SIZE:
            steps = len(buffer) // BATCH_SIZE
            losses_pol, losses_val = [], []
            for _ in range(EPOCHS):
                for _ in range(steps):
                    s, p, z = buffer.sample(BATCH_SIZE)
                    metrics = model.train_on_batch(s, [p, z])
                    losses_pol.append(metrics[1])
                    losses_val.append(metrics[2])
            print(f"\nLoss: Policy={np.mean(losses_pol):.4f} | Value={np.mean(losses_val):.4f}")
            
            # Save checkpoint
            model.save(MODEL_PATH)
            with open(BUFFER_PATH, "wb") as f:
                pickle.dump(buffer, f)
        
        # Arena
        if (iteration + 1) % ARENA_FREQ == 0:
            win_rate = evaluate_vs_best(env, model, BEST_MODEL_PATH, sims=MCTS_SIMS, games=10)
            print(f"Arena Win Rate: {win_rate*100:.1f}%")
            
            if win_rate >= 0.55:
                print("New Champion! Saving best model.")
                model.save(BEST_MODEL_PATH)
        else:
            print("Arena: Skipped (Not an evaluation iteration).")