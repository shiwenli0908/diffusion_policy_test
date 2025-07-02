import numpy as np

def move_to_position(env, target_pos, tolerance=1e-5, max_step=0.05, max_iters=50, render=True):
    obs = env.reset()
    imgs = [env.render(mode='rgb_array')] if render else []
    for step in range(max_iters):
        effector_pos = obs["effector_translation"]
        delta = target_pos - effector_pos
        dist = np.linalg.norm(delta)
        if dist < tolerance:
            #print(f"Reached target {target_pos} in {step} steps.")
            break
        if dist > max_step:
            delta = delta / dist * max_step
        action = np.clip(delta, env.action_space.low, env.action_space.high)
        obs, _, _, info = env.step(action)
        if render:
            imgs.append(env.render(mode='rgb_array'))
    return obs, imgs

# 映射参数
# (0, 0.5) → (0.4, -0.3)
# (1, 0.7) → (0.3, 0.2)
# (1, 0.3) → (0.5, 0.2)
a = (0.3 - 0.4) / (0.7 - 0.5)  # -0.5
b = 0.4 - a * 0.5              # 0.65
c = 0.5
d = -0.3

# 正向变换：从 source 到 target
def transform(point):
    x, y = point
    x_new = a * y + b
    y_new = c * x + d
    return np.array([x_new, y_new])

# 反向变换：从 target 回到 source
def retransform(point_new):
    x_new, y_new = point_new
    y = (x_new - b) / a
    x = (y_new - d) / c
    return np.array([x, y])