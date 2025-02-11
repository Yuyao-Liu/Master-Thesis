import numpy as np
import gym
import inverse_kinematics_gymified
import inverse_kinematics_gymified.envs.forw_kinm
from inverse_kinematics_gymified.envs.inv_kinm import *

#env = gym.make('custom_fetch-v0')
env = gym.make('inverse_kinematics-v0')
#env = gym.make('inverse_kinematics-with-manip-rewards-v0')
#env = gym.make('FetchPickAndPlace-v1')
obs = env.reset() # need to call first 'cos new version errors
env.render()
#obs = env.reset()
done = False
print(env.action_space)
#obs = env._get_obs()

def policy(robot, desired_goal):
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinm_Jac_T(robot, desired_goal)
#    del_thet = invKinm_PseudoInv(robot, desired_goal)
#    del_thet = invKinm_dampedSquares(robot, desired_goal)
    del_thet = invKinm_PseudoInv_half(robot, desired_goal)
#    del_thet = invKinmQP(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kI(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidE_kM(robot, desired_goal)
#    del_thet = invKinmQPSingAvoidManipMax(robot, desired_goal)

    return del_thet

for experiment in range(400):
    print('experiment #:',  experiment)
    print(obs['desired_goal'])
    thetas = []
    for joint in env.robot.joints:
        thetas.append(joint.theta)
    print("POST:", thetas)
    for step in range(50):
        env.render()
        #action = policy(obs['observation'], obs['desired_goal'])
        action = policy(env.robot, obs['desired_goal'])
        #obs, reward, done, info = env.step(action)
        action_fix = list(action)
        action_fix.append(1.0)
        action_fix = np.array(action_fix)
        #print(type(action[0]), action)
        #print(type(list(action)[0]))
        #print(list(action) + [1.0])
        #print(type(action_fix), action_fix)
        obs, reward, done, info = env.step(action_fix)
    thetas = []
    for joint in env.robot.joints:
        thetas.append(joint.theta)
    obs = env.reset_test()
    print("PRE:", thetas)


