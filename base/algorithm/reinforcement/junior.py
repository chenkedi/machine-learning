# coding=utf-8

import numpy as np
import pandas as pd
import time

np.random.seed(2)
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
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform() > gamma) or (state_actions.all() == 0):
        action_name = np.random.choice(actions)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == init_states - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(init_states-1) + ['T']   # '---------T' our environment
    # S对应环境env_list的索引，范围为0-4，如果为terminal，则表示到达目标
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(gap_time)
        print('\r                               ', end='')  # 这里是抹除当前打印行的信息
    else:
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
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]  # 注意这里是S，不是_
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

