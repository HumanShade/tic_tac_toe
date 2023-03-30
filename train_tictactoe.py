import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def play(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.player
            self.player = -self.player
            return True
        return False

    def is_game_over(self):
        for i in range(3):
            if abs(self.board[i].sum()) == 3 or abs(self.board[:, i].sum()) == 3:
                return True
            if abs(self.board.diagonal().sum()) == 3 or abs(np.fliplr(self.board).diagonal().sum()) == 3:
                return True
        return (self.board == 0).sum() == 0

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def optimize_model(memory, policy_net, target_net, optimizer):
    state, action, reward, next_state, done = memory
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    action = torch.tensor([action], dtype=torch.int64).unsqueeze(0).to(device)
    reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
    done = torch.tensor([int(done)], dtype=torch.bool).unsqueeze(0).to(device)

    predicted_q = policy_net(state).gather(1, action)
    next_q = torch.zeros(1).to(device)
    if not done.item():
        next_q = target_net(next_state).max(1)[0].detach()
    expected_q = reward + GAMMA * next_q.unsqueeze(1)

    loss = nn.MSELoss()(predicted_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def select_action(policy_net, state, device):
    with torch.no_grad():
        q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        q_values = q_values.cpu().numpy()
        valid_actions = np.where(state == 0)[0]
        return valid_actions[np.argmax(q_values[0][valid_actions])]


def train(policy_net, target_net, optimizer, device, epochs=10000):
    for epoch in tqdm(range(epochs), desc="Training status"):
        game = TicTacToe()
        while not game.is_game_over():
            state = game.board.flatten() * game.player
            action = select_action(policy_net, state, device)
            row, col = action // 3, action % 3
            game.play(row, col)
            reward = -1 if game.is_game_over() else 0
            next_state = game.board.flatten() * game.player
            memory = (state, action, reward, next_state, game.is_game_over())
            optimize_model(memory, policy_net, target_net, optimizer)
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print("Training finished")


if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.9
    TARGET_UPDATE = 100

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks and optimizer
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters())

    # Train the network
    train(policy_net, target_net, optimizer, device)

    # Save the trained model
    torch.save(policy_net.state_dict(), "tictactoe_dqn.pt")

