from rl_pacman.environment import PacmanEnv
import cv2
import random


FPS = 30


class RandomPlay():

    def __init__(self) -> None:
        self.__pacman_env = PacmanEnv()

    def __random_policy(self, state):
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
        states, lives, actions, rewards = self.__pacman_env.play_game(self.__random_policy)
        if save_animation:
            self.__save_animations(states, save_path)
        return states, lives, actions, rewards
    
    def stop(self) -> None:
        self.__pacman_env.stop()



def main():
    # Create Environment
    random_play = RandomPlay()

    # Play Game
    random_play.play_game(r"C:\Users\revue\Documents\rl_pacman\animations\random_game.avi", True)
    random_play.stop()


if __name__ == "__main__":
    main()
    