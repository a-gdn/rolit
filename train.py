import random
import math
import numpy as np
import tensorflow as tf
from collections import deque, defaultdict
import os
import pickle
import subprocess
import atexit
import argparse

# Import common game logic and MCTS implementation
from common import (
    RolitEnv, 
    MCTS, 
    preprocess_state, 
    build_model, 
    get_next_player, 
    BOARD_SIZE, 
    L2_REG,
    DEFAULT_NUM_PLAYERS
)

# ==========================================
# 0. SYSTEM SETUP
# ==========================================
# Prevent macOS sleep during training (optional, harmless on Linux/Windows)
try:
    caffeinate_proc = subprocess.Popen(['caffeinate', '-dimsu'])
    atexit.register(lambda: caffeinate_proc.terminate())
    print("System sleep prevention enabled.")
except Exception:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CONFIGURATION
# ==========================================
LEARNING_RATE = 0.001

# TRAINING HYPERPARAMETERS
ITERATIONS = 100         
EPISODES_PER_ITER = 10   
MCTS_SIMS = 400          
BATCH_SIZE = 256
EPOCHS = 3
BUFFER_SIZE = 100000 
RESIDUAL_BLOCKS = 5      
CPUCT = 2.0              

# MCTS EXPLORATION HYPERPARAMETERS (Dirichlet Noise)
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25 

# FILE PATHS
MODEL_PATH_BASE = "rolit_model"      
BEST_MODEL_PATH_BASE = "best_rolit_model"
BUFFER_PATH_BASE = "replay_buffer"
LOG_DIR_BASE = "logs/rolit_training"

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
            # Check for game end
            if env.done: break 
            
            # Must pass
            player = get_next_player(player, env.num_players)
            continue

        # Run MCTS
        pi = mcts.run(
            board, 
            player, 
            env.move_count, 
            is_self_play=True,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON
        )
        trajectory.append((board.copy(), pi, player, env.move_count))
        
        # Apply temperature (exploration) for the first few moves
        if env.move_count < 12: 
            # Sample from the policy vector
            action = np.random.choice(len(pi), p=pi)
        else:
            # Select the move with the highest visit count (deterministic)
            action = np.argmax(pi)
            
        r, c = divmod(action, env.board_size)
        
        # Take the step
        env.step((r, c), player)
        board = env.board.copy() # Update board for next loop
        
        if env.done: break
        player = get_next_player(player, env.num_players)
        
    winner = env.winner()
    data = []
    
    # Process trajectory for training
    for s, pi, turn, mc in trajectory:
        # Calculate result (z) relative to the player 'turn'
        if winner == 0: z = 0
        else: z = 1 if winner == turn else -1
            
        state = preprocess_state(s, turn, mc, env.num_players) 
        
        # 8-Way Augmentation (only for 8x8 square boards)
        if env.board_size == 8:
            pi_board = pi.reshape(BOARD_SIZE, BOARD_SIZE)
            for k in range(4):
                rot_state = np.rot90(state, k)
                rot_pi = np.rot90(pi_board, k)
                data.append((rot_state, rot_pi.flatten(), z))
                
                # Flip Left/Right
                data.append((np.fliplr(rot_state), np.fliplr(rot_pi).flatten(), z))
        else:
            data.append((state, pi, z)) # No augmentation for non-square boards

    return data

# ==========================================
# 3. ARENA EVALUATION (MODEL VS MODEL)
# ==========================================
def evaluate_vs_best(env, current_model, best_model_path, sims, games):
    if not os.path.exists(best_model_path):
        return 1.0 # Current is champion by default if no history

    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    
    wins = 0
    draws = 0
    
    for i in range(games):
        print(f"Arena: Pitting Current vs Best Model, Game {i + 1}/{games}...", end='\r')

        # Swap sides: Current model plays P1, P2, P3, ... in a cycle
        start_player_current = (i % env.num_players) + 1
        
        models = {}
        for p in range(1, env.num_players + 1):
            if p == start_player_current:
                models[p] = current_model
            else:
                models[p] = best_model # All other opponents use the best model
        
        env.reset()
        board = env.board.copy()
        player = 1
        
        mcts_agents = {p: MCTS(env, models[p], simulations=sims, num_players=env.num_players) for p in range(1, env.num_players + 1)}
        
        while not env.done:
            legal = env.get_legal_moves(player)
            if not legal:
                if env.done: break
                player = get_next_player(player, env.num_players)
                continue
            
            # Select Move (Deterministic - No Noise)
            pi = mcts_agents[player].run(board, player, env.move_count, is_self_play=False)
                
            action = np.argmax(pi)
            r, c = divmod(action, env.board_size)
            
            env.step((r, c), player)
            board = env.board.copy()
            player = get_next_player(player, env.num_players)

        winner = env.winner()
        
        # Calculate result for CURRENT model
        if winner == 0:
            draws += 1
        # Check if the current model was the winner
        elif winner == start_player_current: 
            wins += 1
    
    win_rate = (wins + 0.5 * draws) / games
    return win_rate

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent for Rolit using MCTS and a neural network.')
    parser.add_argument('--num_players', type=int, default=DEFAULT_NUM_PLAYERS, choices=[2, 3, 4], help='Number of players in the game.')
    args = parser.parse_args()
    
    NUM_PLAYERS = args.num_players
    
    # Dynamic file paths based on player count
    MODEL_PATH = f"{MODEL_PATH_BASE}_{NUM_PLAYERS}p.keras"
    BEST_MODEL_PATH = f"{BEST_MODEL_PATH_BASE}_{NUM_PLAYERS}p.keras"
    BUFFER_PATH = f"{BUFFER_PATH_BASE}_{NUM_PLAYERS}p.pkl"
    LOG_DIR = f"{LOG_DIR_BASE}_{NUM_PLAYERS}p"
    
    env = RolitEnv(BOARD_SIZE, NUM_PLAYERS)
    summary_writer = tf.summary.create_file_writer(LOG_DIR)
    
    # The number of input planes for the NN is (num_players + 1)
    INPUT_PLANES = NUM_PLAYERS + 1 
    
    # 1. LOAD BUFFER
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, "rb") as f:
            buffer = pickle.load(f)
        print(f"Loaded buffer for {NUM_PLAYERS}P game with {len(buffer)} samples.")
    else:
        buffer = ReplayBuffer(BUFFER_SIZE)
        print(f"Created new buffer for {NUM_PLAYERS}P game.")

    # 2. LOAD/BUILD MODEL
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Re-compile to ensure correct learning rate and optimizer state
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
                      loss={'policy': 'categorical_crossentropy', 'value': 'mse'})
        print(f"Loaded existing model for {NUM_PLAYERS}P game.")
    else:
        model = build_model(INPUT_PLANES, RESIDUAL_BLOCKS, LEARNING_RATE)
        print(f"Created new model for {NUM_PLAYERS}P game.")

    # 3. INITIALIZE BEST MODEL
    if not os.path.exists(BEST_MODEL_PATH):
        model.save(BEST_MODEL_PATH)
        print("Initialized Best Model snapshot.")

    print(f"Starting Training: {ITERATIONS} Iterations for {NUM_PLAYERS} Players")

    for iteration in range(ITERATIONS):
        print(f"\n--- Iteration {iteration+1}/{ITERATIONS} ---")
        
        # A. SELF PLAY
        new_samples = 0
        for i in range(EPISODES_PER_ITER):
            data = self_play(env, model, MCTS_SIMS) 
            for sample in data:
                buffer.add(*sample)
            new_samples += len(data)
            print(f"Self-Play {i+1}/{EPISODES_PER_ITER}: Buffer={len(buffer)}", end='\r')
        print(f"\nCollected {new_samples} new samples.")
        
        # B. TRAINING
        avg_loss, avg_val_loss = 0, 0
        if len(buffer) > BATCH_SIZE * EPOCHS: # Ensure enough data for full epochs
            losses_total, losses_pol, losses_val = [], [], []
            steps_per_epoch = len(buffer) // BATCH_SIZE
            print(f"Training {EPOCHS} epochs over {steps_per_epoch} steps each...")
            
            for _ in range(EPOCHS):
                for _ in range(steps_per_epoch):
                    s, p, z = buffer.sample(BATCH_SIZE)
                    metrics = model.train_on_batch(s, [p, z])
                    losses_total.append(metrics[0])
                    losses_pol.append(metrics[1])
                    losses_val.append(metrics[2])
            
            avg_loss = np.mean(losses_total)
            avg_val_loss = np.mean(losses_val)
            print(f"Loss: Total={avg_loss:.4f} | Policy={np.mean(losses_pol):.4f} | Value={avg_val_loss:.4f}")
        else:
            print(f"Buffer size {len(buffer)} too small to train with Batch Size {BATCH_SIZE} * Epochs {EPOCHS}.")

        # C. EVALUATION (ARENA)
        win_rate = evaluate_vs_best(env, model, BEST_MODEL_PATH, sims=MCTS_SIMS, games=10)
        print(f"Arena Win Rate (vs Champion): {win_rate * 100:.1f}%")
        
        # Update Champion if we are significantly better (55% threshold)
        if win_rate >= 0.55:
            print("New Champion! Updating the best model")
            model.save(BEST_MODEL_PATH)
        else:
            print("Current model failed to beat the best model")

        # D. LOGGING & SAVING
        model.save(MODEL_PATH)
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(buffer, f)
            
        with summary_writer.as_default():
            tf.summary.scalar('loss/total', avg_loss, step=iteration)
            tf.summary.scalar('loss/value', avg_val_loss, step=iteration)
            tf.summary.scalar('win_rate/arena', win_rate, step=iteration)