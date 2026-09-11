def move_toward_target(data, target, speed=0.02):
    for i in range(len(target)):
        difference = target[i] - data.qpos[i]

        if abs(difference) < speed:
            data.qpos[i] = target[i]
        elif difference > 0:
            data.qpos[i] += speed
        else:
            data.qpos[i] -= speed