# 这个例子与找宝藏不一样的地方在于Q table的维度需要实时更新
# 假如出现了没出现过的state，需要加入新的一行
# state可以理解为一个坐标，具体在maze_env.py中的step函数中的 s = self.canvas.coords(self.rect)
# Q table的形式为
#          Left   Right   Down   Up
# state1   xxx     xxx    xxx    xxx
# state2   xxx     xxx    xxx    xxx
# ...      xxx     xxx    xxx    xxx
# statex   xxx     xxx    xxx    xxx

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    # 学习 100 回合
    for episode in range(100):
        # 初始化 state 的观测值
        observation = env.reset()

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)

            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将下一个 state 的值传到下一次循环
            observation = observation_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                break

    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()