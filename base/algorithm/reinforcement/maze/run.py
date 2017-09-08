from base.algorithm.reinforcement.maze.maze_env import Maze
from rl_brain import QLearningTable

max_iteration = 100


def update():
    for iter in range(max_iteration):
        # 初始化 state 的观测值
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(max_iteration, update)
    env.mainloop()
