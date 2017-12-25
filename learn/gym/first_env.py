import gym


def create(env_name):
    env = gym.make(env_name)
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())


def create_advanced(env_name):
    env = gym.make(env_name)
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == "__main__":
    name1 = 'CartPole-v0'
    name2 = 'MountainCar-v0'
    name3 = 'MsPacman-v0'

    # create(name2)
    create_advanced(name1)
