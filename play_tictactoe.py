import torch
import numpy as np
from pathlib import Path
from train_tictactoe import DQN, TicTacToe, select_action

class HumanPlayer:
    def get_move(self, state):
        while True:
            move = input("Enter your move (row and column separated by space): ")
            row, col = map(int, move.strip().split())
            if state[row][col] == 0:
                return row, col
            else:
                print("Invalid move. Please try again.")

def load_model(model_path, device):
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def print_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    for row in board:
        print('|'.join([symbols[val] for val in row]))
        print('-' * 5)

def play_game(player1, player2, model, device):
    game = TicTacToe()
    players = [player1, player2]
    while not game.is_game_over():
        print("\nCurrent board state:")
        print_board(game.board)
        player = game.player
        if player == 1:
            row, col = players[player - 1].get_move(game.board)
        else:
            state = game.board.flatten() * player
            action = select_action(model, state, device)
            row, col = action // 3, action % 3
        game.play(row, col)

    print("\nFinal board state:")
    print_board(game.board)
    winner = "Draw" if game.board.sum() == 0 else "Player 1" if game.player == -1 else "Player 2"
    print(f"Game over. Winner: {winner}")



if __name__ == "__main__":
    model_path = Path("tictactoe_dqn.pt")
    if not model_path.is_file():
        print("Trained model not found. Please run the training script first.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device)
        player1 = HumanPlayer()
        player2 = None  # The AI player will be using the model directly

        print("Tic Tac Toe: You (Player 1) vs. AI (Player 2)")
        play_game(player1, player2, model, device)
