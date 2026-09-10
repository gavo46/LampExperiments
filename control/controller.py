def move_toward_target(data, target, smoothing=0.05): 
    """ Gradually move each robot joint toward target. """ 
    for i in range(len(target)): 
        current = data.qpos[i] 
        difference = target[i] - current 
        data.qpos[i] = current + smoothing * difference