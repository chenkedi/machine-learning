# coding=utf-8

import numpy as np
import pandas as pd
import time

np.random.seed(1)
init_states = 6
actions = ['left', 'right']
epsilon = 0.9
alpha = 0.1
gamma = 0.9
max_iteration = 13
fresh_time = 0.2  # 每个方案内部Q值迭代的间隔
gap_time = 1  # 每个方案迭代的间隔


def build_q_table(init_states, actions):
    table = pd.DataFrame(
        np.zeros((init_states, len(actions))),  # 以状态为行，动作为列，交点的值表示该状态下选择该动作的Q值
        columns=actions
    )
    return table


def choose_action(state, q_table):
    """
    基于epsilon贪心的策略Pi
    :param state:
    :param q_table:
    :return:
    """
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > epsilon) or (state_actions.multi_thread_retriver() == 0):
        action_name = np.random.choice(actions)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == init_states - 2:  # 当当前选择的动作为向右，且当前位置为终点的前一个点，则转以后的状态S_为terminal
            S_ = 'terminal'
            R = 1  # 到达终点reward为1
        else:
            S_ = S + 1  # 如果位置不是倒数第一个，则位置加一
            R = 0
    else:
        R = 0  # 向左走reward都为0
        if S == 0:  # 本身处于第0个位置，向左走还是在这个位置
            S_ = S
        else:
            S_ = S - 1  # 否则位置减一
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (init_states - 1) + ['T']  # '---------T' our environment
    # S对应环境env_list的索引，范围为0-4，如果为terminal，则表示到达目标
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(gap_time)
        print('\r                               ', end='')  # 这里是抹除当前打印行的信息
    else:  # 如果没到终点，则打印出根据Q值选择的策略在环境中得到的状态
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(fresh_time)


def reinforcement_learning():
    q_table = build_q_table(init_states, actions)

    for iter in range(max_iteration):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, iter, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)  # 使用epsilon贪心策略选择动作
            S_, R = get_env_feedback(S, A)  # 根据选择的动作和当前所处的状态获得环境的反馈，得到奖励和下一个状态
            q_predict = q_table.ix[S, A] # 获得上一个状态和所选择动作的Q值
            if S_ != 'terminal':
                q_target = R + gamma * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.ix[S, A] += alpha * (q_target - q_predict)
            S = S_

            update_env(S, iter, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = reinforcement_learning()
    print('\r\nQ-table:\n')
    print(q_table)
