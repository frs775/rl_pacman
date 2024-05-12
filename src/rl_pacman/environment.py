"""
Pacman Environment
"""

import gym


class PacmanEnv():

    def __init__(self) -> None:
        # Environment
        self.__env =  gym.make("ALE/MsPacman-v5")
        # Atributes
        self.__action_space = self.__env.action_space
        self.__state = None
        self.__lives = None
        self.__done = False
        self.reset()

    def reset(self) -> None:
        state, info = self.__env.reset()
        self.__state = state
        self.__lives = info["lives"]
        self.__done = False

    def move(self, action: int) -> float:
        state, reward, done, _, info = self.__env.step(action)
        self.__state = state
        self.__lives = info["lives"]
        self.__done = done
        return reward

    def play_game(self, policy) -> tuple:
        # Reset Game
        self.reset()
        # Variables
        states = [self.__state]
        lives = [self.__lives]
        actions = []
        rewards = []
        # Play Following Policy
        while not self.__done:
            # Take action
            action = policy(self.__state)
            reward = self.move(action)
            # Update Variables
            states.append(self.__state)
            lives.append(self.__lives)
            actions.append(action)
            rewards.append(reward)
        return states, lives, actions, rewards

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