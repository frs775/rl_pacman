from rl_pacman.environment import PacmanEnv
from rl_pacman.q_learning.q_matrix import QMatrix
import cv2
import random
import numpy as np


FPS = 30


class DoubleQLearning():

    def __init__(self) -> None:
        # Create Environment
        self.__pacman_env = PacmanEnv()
        # Q Matrix (Sparse)
        self.__Q1 = QMatrix()
        self.__Q2 = QMatrix()

    def train_double_q_learning(self, num_games: int = 10, alpha: float = 0.5, gamma: float = 0.5) -> None:
        # For each Game
        for _ in range(num_games):
            # Start Game
            self.__pacman_env.reset()
            # While Game is On
            while not self.__pacman_env.done:
                # Choose Action
                origin_state = frozenset(self.__pacman_env.state.reshape(-1,))
                action = self.__best_action(origin_state)
                # Move
                reward = self.__pacman_env.move(action)
                new_state = frozenset(self.__pacman_env.state.reshape(-1,))
                # Update Q
                if random.randint(0, 1):
                    new_value = self.__Q1.get(origin_state, action) + alpha * (reward + gamma * (self.__best_action_Q2(new_state)) - self.__Q1.get(origin_state, action))
                    self.__Q1.set(origin_state, action, new_value)
                else:
                    new_value = self.__Q2.get(origin_state, action) + alpha * (reward + gamma * (self.__best_action_Q1(new_state)) - self.__Q2.get(origin_state, action))
                    self.__Q2.set(origin_state, action, new_value)
    
    def __double_q_policy(self, state) -> int:
        # Unmutable to Hash State
        hash_state = frozenset(state.reshape(-1,))
        # Get Best Action
        action = self.__best_action(hash_state)
        return action
    
    def __best_action(self, hash_state) -> int:
        # Choose Action
        scores = list(map(lambda action:
                          self.__Q1.get(hash_state, action) + self.__Q2.get(hash_state, action),
                          range(self.__pacman_env.action_space.n)
                      )
                 )
        action_scores = dict(zip(range(self.__pacman_env.action_space.n), scores))
        action = max(action_scores, key=action_scores.get)
        # If not information for the state (all actions < max = 0) -> random policy 
        if not action_scores[action]:
            action = random.randint(0, self.__pacman_env.action_space.n - 1)
        return action
    
    def __best_action_Q1(self, hash_state) -> int:
        # Choose Action
        scores = list(map(lambda action:
                          self.__Q1.get(hash_state, action),
                          range(self.__pacman_env.action_space.n)
                      )
                 )
        action_scores = dict(zip(range(self.__pacman_env.action_space.n), scores))
        action = max(action_scores, key=action_scores.get)
        # If not information for the state (all actions < max = 0) -> random policy 
        if not action_scores[action]:
            action = random.randint(0, self.__pacman_env.action_space.n - 1)
        return action
    
    def __best_action_Q2(self, hash_state) -> int:
        # Choose Action
        scores = list(map(lambda action:
                          self.__Q2.get(hash_state, action),
                          range(self.__pacman_env.action_space.n)
                      )
                 )
        action_scores = dict(zip(range(self.__pacman_env.action_space.n), scores))
        action = max(action_scores, key=action_scores.get)
        # If not information for the state (all actions < max = 0) -> random policy 
        if not action_scores[action]:
            action = random.randint(0, self.__pacman_env.action_space.n - 1)
        return action

    def __save_animations(self, states, save_path: str) -> None:
        height, width, layers = states[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))
        for image in states:
            # Convert the image from RGB to BGR, which is what OpenCV expects
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image_bgr)
        # Release everything when job is finished
        out.release()

    def play_game(self, save_path: str, save_animation: bool = False) -> tuple:
        states, lives, actions, rewards = self.__pacman_env.play_game(self.__double_q_policy)
        if save_animation:
            self.__save_animations(states, save_path)
        return states, lives, actions, rewards
    
    def stop(self) -> None:
        self.__pacman_env.stop()


def main():
    q_learnig = DoubleQLearning()
    q_learnig.train_double_q_learning(1, 0.8, 0.8)
    q_learnig.play_game(r"C:\Users\revue\Documents\rl_pacman\animations\double_q_learning_game.avi", True)
    q_learnig.stop()


if __name__ == "__main__":
    main()
