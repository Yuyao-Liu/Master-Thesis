# pixel trajectory: DONE
-------------------------
have your pixels in a 2D array of length N points : ((x0,y0), (x1,y1),...(xN,yN))
time is just np.arange(N).
these guys are relatively sparse, ~150 points. 
in the program the distribution is dependent on how fast you move your mouse,
which is why i assume means it's not of great importance as long as it
covers everything reasonably.
you might want to look into bezier-ing the path into something smooth.
but looks OK as is also

# 3D cartesian trajectory: DONE
-------------------------
just put your pixels onto a plane onto a predefined place with the right coordinate transform.
ofc you have the pixel -> mm conversion and mins and maxes. whatever makes sense workspace-wise is good.
- wrote an algorithm to automatically calibrate where the board is
TODO: 
- i manually set signs of the rotation matrix rn, would be nice if this was consistent
  (also fck rpy, but that's what i have access to)

# timed joint trajectory: DONE
-----------------------------
- assumed 10s 'cos who cares
- the dmp class code handles the finite differencing
the UR matlab module just magically calculates what's needed.
thus just running clik along the points to get joint angles for each point is almost certainly fine,
although can be probably made better or worse.
once you got q(t), you just do the most basic finite-difference 
to get q_dot(t) and q_ddot(t). 
that's it as far as trajectory generation is concerned.
the result in the end is a 
struct q_traj{
	t : 1 x N;
	pos: 6 x N;
	vel: 6 x N;
	acc: 6 x N
}
how to read this from the .mat if you leave it this way idk.
but you'll almost certainly want to replicate this code anyway so who cares.

# dmp-ing the trajectory - DONE (albin's code gg ez)
--------------------------
- TODO: understand what's going on
- idea type beat: we're doing it in joint space, but since it's just a demo,
doing it with speedl might be easier (it's not like 
the damped pseudoinverse is peak performance, let's be real)


# control loop around trajectory following - DONE within dmp
-------------------------------------------
- TODO for fun this with alternative basic methods to get a better grip on things
- you'll need to measure and analyse stuff in a non-trivial way tho
it's literally just feedforward position and feedback velocity
```python
def compute_ctrl(self, target_state, robot_state):
    ff = 0.
    fb = 0.
    target_position = np.array(target_state.position)
    target_velocity = np.array(target_state.velocity)
    robot_position = np.array(robot_state.position)
    if target_velocity.size > 0:
        ff = target_velocity
    if target_position.size > 0:
        fb = self.kp * (target_position - robot_position)
    vel_cmd = ff + fb
    np.clip(vel_cmd, -self.vel_max, self.vel_max, out=vel_cmd)

    return vel_cmd
```

# add some impedance to live with errors - DONE, but needs filtering
--------------------------------------
- f/t readings are added and works fine for impedance
- TODO: cancel out the gripper weight -> DONE, but TODO put the payload
  assignment into code to avoid surprises

# make a main file that runs all of it at once
----------------------------------------
- do it
