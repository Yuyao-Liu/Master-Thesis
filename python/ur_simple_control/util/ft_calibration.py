
def calibrateFT(robot):
    ft_readings = []
    for i in range(2000):
        start = time.time()
        q = robot.rtde_receive.getActualQ()
        ft = robot.rtde_receive.getActualTCPForce()
        tau = robot.rtde_control.getJointTorques()
        current = robot.rtde_receive.getActualCurrent()
        ft_readings.append(ft)
        end = time.time()
        diff = end - start
        if diff < robot.dt:
            time.sleep(robot.dt - diff)

    ft_readings = np.array(ft_readings)
    avg = np.average(ft_readings, axis=0)
    print("average ft time", avg)
    return avg
