"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

这个例子与找宝藏不一样的地方在于Q table的维度需要实时更新
假如出现了没出现过的state，需要加入新的一行
state可以理解为一个坐标，具体在maze_env.py中的step函数中的 s = self.canvas.coords(self.rect)
Q table的形式为
         Left   Right   Down   Up
state1   xxx     xxx    xxx    xxx
state2   xxx     xxx    xxx    xxx
...      xxx     xxx    xxx    xxx
statex   xxx     xxx    xxx    xxx
其中state类型为坐标（类似），action类型为数字，0代表up···
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # print
            # print(RL.q_table)

            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()