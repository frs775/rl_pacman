import random

class QMatrix():

    def __init__(self) -> None:
        self.__matrix = {}

    def get(self, state, action) -> float:
        value = 0
        if state in self.__matrix:
            if action in self.__matrix[state]:
                value = self.__matrix[state][action]
        return value
        
    def set(self, state, action, value) -> None:
        if state in self.__matrix:
            self.__matrix[state][action] = value
        else:
            self.__matrix[state] = {action: value}