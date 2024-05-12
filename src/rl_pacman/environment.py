"""
Pacman Environment
"""

import gym
import matplotlib.pyplot as plt


class PacmanEnv():

    def __init__(self) -> None:
        # Environment
        self.__env =  gym.make('ALE/MsPacman-v5', render_mode='human')
        # Atributes
        self.__action_space = self.__env.action_space
        self.__state = None
        self.__rewards = None
        self.__lives = None
        self.__done = False
        self.reset()

    def display(self) -> None:
        plt.imshow(self.__state)
        plt.title(f"Lives {self.__lives} - Reward: {sum(self.__rewards)}")
        plt.axis('off')
        plt.show()

    def reset(self) -> None:
        state, info = self.__env.reset()
        self.__state = state
        self.__lives = info["lives"]
        self.__rewards = []
        self.__done = False

    def move(self, action: int) -> None:
        if action not in self.__action_space:
            raise ValueError("Not available action. Avalilable actions: [0, 1, ..., 8]")
        state, reward, done, _, info = self.__env.step(action)
        self.__state = state
        self.__rewards.append(reward)
        self.__lives = info["lives"]
        self.__done = done

    def stop(self) -> None:
        self.__env.close()

    @property
    def lives(self) -> int:
        return self.__lives
    
    @property
    def action_space(self) -> int:
        return self.__action_space
    
    @property
    def state(self):
        return self.__state
    
    @property
    def done(self) -> bool:
        return self.__done