from rl_pacman.environment import PacmanEnv
import cv2
import random


FPS = 30


class QLearning():

    def __init__(self) -> None:
        # Create Environment
        self.__pacman_env = PacmanEnv()
        # Q Matrix (Sparse)
        self.__Q = {}

    def train_q_learning(self, num_games: int = 10, alpha: float = 0.5, gamma: float = 0.5) -> None:
        # For each Game
        for _ in range(num_games):
            # Start Game
            self.__pacman_env.reset()
            # While Game is On
            while not self.__pacman_env.done:
                # Choose Action
                origin_state = frozenset(self.__pacman_env.state.reshape(-1,))
                if (origin_state in self.__Q) and self.__Q[origin_state]:
                    action = max(self.__Q[origin_state], key=self.__Q[origin_state].get)
                else:
                    action = random.randint(0, self.__pacman_env.action_space.n - 1)
                    self.__Q[origin_state] = {action: 0}
                # Move
                reward = self.__pacman_env.move(action)
                # Update Q
                # max
                new_state = frozenset(self.__pacman_env.state.reshape(-1,))
                if (new_state in self.__Q) and self.__Q[new_state]:
                    max_new_state = max(self.__Q[new_state], key=self.__Q[new_state].get)
                else:
                    max_new_state = 0
                # Value
                self.__Q[origin_state][action] = self.__Q[origin_state][action] + alpha * (reward + gamma * (max_new_state) - self.__Q[origin_state][action])

    def __q_policy(self, state) -> int:
        hash_state = frozenset(state.reshape(-1,))
        # If visited state during training
        if (hash_state in self.__Q):
            if self.__Q[hash_state]:
                return  max(self.__Q[hash_state], key=self.__Q[hash_state].get)
        else:
            return random.randint(0, self.__pacman_env.action_space.n - 1)
        
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
        states, lives, actions, rewards = self.__pacman_env.play_game(self.__q_policy)
        if save_animation:
            self.__save_animations(states, save_path)
        return states, lives, actions, rewards
    
    def stop(self) -> None:
        self.__pacman_env.stop()


def main():
    q_learnig = QLearning()
    q_learnig.train_q_learning(1, 0.8, 0.8)
    q_learnig.play_game(r"C:\Users\revue\Documents\rl_pacman\animations\q_learning_game.avi", True)
    q_learnig.stop()


if __name__ == "__main__":
    main()
