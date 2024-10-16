

env = OpenAIEnvrionment("Walker2d-v5", param=False)
env.set_seed(10)

for i in range(1000):
    state = env.reset()
    for j in range(1000):
        action = env.sample_action()
        next_state, reward, done, info = env.step(action)

        recon_rwd = get_openai_walker_reward(state, action, next_state)

        if not (recon_rwd == reward):
            print("Different")
            print(recon_rwd)
            print(reward)

        state = next_state


