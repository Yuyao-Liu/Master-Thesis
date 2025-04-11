#########################################################3
# for now only for ur10e, later write out for other robots
#########################################################3
import numpy as np


def getAllMotors(robot):
    motors = []
    motors.append(robot.getMotor('shoulder_pan_joint'))
    motors.append(robot.getMotor('shoulder_lift_joint'))
    motors.append(robot.getMotor('elbow_joint'))
    motors.append(robot.getMotor('wrist_1_joint'))
    motors.append(robot.getMotor('wrist_2_joint'))
    motors.append(robot.getMotor('wrist_3_joint'))
    print("imported motors")
    return motors


# for validation purposes, we must plot this circle
def drawCircle(radius, height, robot):
    display = robot.getDisplay("display")
#    display.setColor(0xFFFFF)
    display.setColor(0xF8FF04)
    display.setColor(0xF8FF04)
    display.setColor(0xFF0022)
    display.setColor(0xFF0022)
    display.setColor(0x00FFE6)
    display.setColor(0x00FFE6)
    display.drawOval(100,100,60,60)
    display.drawOval(100,100,60,60)
    display.drawOval(100,100,59,59)
    display.drawOval(100,100,59,59)
    print("drew the circle on the screen")

def setMotorSpeeds(motors, speeds):
    for i in range(len(motors)):
        motors[i].setVelocity(speeds[i])



def initializeMotorsForVelocity(motors):
    speeds = []
    for motor in motors:
        motor.setPosition(float('inf'))
        speeds.append(0)

    
    setMotorSpeeds(motors, speeds)



def initializeMotorsForPosition(motors):
    speeds = []
    for motor in motors:
#        motor.setVelocity(float('inf'))
        speeds.append(0)

    
    setMotorSpeeds(motors, speeds)
    print("did motor init")


def getAndInitAllSensors(robot):
    sensors = []
    sensors.append(robot.getPositionSensor('shoulder_pan_joint_sensor'))
    sensors.append(robot.getPositionSensor('shoulder_lift_joint_sensor'))
    sensors.append(robot.getPositionSensor('elbow_joint_sensor'))
    sensors.append(robot.getPositionSensor('wrist_1_joint_sensor'))
    sensors.append(robot.getPositionSensor('wrist_2_joint_sensor'))
    sensors.append(robot.getPositionSensor('wrist_3_joint_sensor'))
   
   # now we must specify how often do sensors get read in miliseconds
   # i've put 10ms since this seems like it should work smoothly,
   # but ofc you should play with this value later on
    for sensor in sensors:
        sensor.enable(10)

    print("imported and inited sensors")
    return sensors

def readJointState(sensors):
    joint_positions = []
    for sensor in sensors:
        joint_positions.append(sensor.getValue())
    return joint_positions





def getAllMotorsJaco(robot):
    motors = []
    motors.append(robot.getMotor('j2n6s300_joint_1'))
    motors.append(robot.getMotor('j2n6s300_joint_2'))
    motors.append(robot.getMotor('j2n6s300_joint_3'))
    motors.append(robot.getMotor('j2n6s300_joint_4'))
    motors.append(robot.getMotor('j2n6s300_joint_5'))
    motors.append(robot.getMotor('j2n6s300_joint_6'))
#    motors.append(robot.getMotor('j2n6s300_joint7'))
    print("imported motors")
    return motors


def getAndInitAllSensorsJaco(robot):
    sensors = []
    sensors.append(robot.getPositionSensor('j2n6s300_joint_1_sensor'))
    sensors.append(robot.getPositionSensor('j2n6s300_joint_2_sensor'))
    sensors.append(robot.getPositionSensor('j2n6s300_joint_3_sensor'))
    sensors.append(robot.getPositionSensor('j2n6s300_joint_4_sensor'))
    sensors.append(robot.getPositionSensor('j2n6s300_joint_5_sensor'))
    sensors.append(robot.getPositionSensor('j2n6s300_joint_6_sensor'))
#    sensors.append(robot.getPositionSensor('j2n6s300_joint7_sensor'))
   
   # now we must specify how often do sensors get read in miliseconds
   # i've put 10ms since this seems like it should work smoothly,
   # but ofc you should play with this value later on
    for sensor in sensors:
        sensor.enable(10)

    print("imported and inited sensors")
    return sensors




def getAllMotorsKuka(robot):
    motors = []
    motors.append(robot.getMotor('joint_a1'))
    motors.append(robot.getMotor('joint_a2'))
    motors.append(robot.getMotor('joint_a3'))
    motors.append(robot.getMotor('joint_a4'))
    motors.append(robot.getMotor('joint_a5'))
    motors.append(robot.getMotor('joint_a6'))
    motors.append(robot.getMotor('joint_a7'))
    print("imported motors")
    return motors


def getAndInitAllSensorsKuka(robot):
    sensors = []
    sensors.append(robot.getPositionSensor('joint_a1_sensor'))
    sensors.append(robot.getPositionSensor('joint_a2_sensor'))
    sensors.append(robot.getPositionSensor('joint_a3_sensor'))
    sensors.append(robot.getPositionSensor('joint_a4_sensor'))
    sensors.append(robot.getPositionSensor('joint_a5_sensor'))
    sensors.append(robot.getPositionSensor('joint_a6_sensor'))
    sensors.append(robot.getPositionSensor('joint_a7_sensor'))
   
   # now we must specify how often do sensors get read in miliseconds
   # i've put 10ms since this seems like it should work smoothly,
   # but ofc you should play with this value later on
    for sensor in sensors:
        sensor.enable(10)

    print("imported and inited sensors")
    return sensors



def getAllMotorsJaco7(robot):
    motors = []
    motors.append(robot.getMotor('j2s7s300_joint_1'))
    motors.append(robot.getMotor('j2s7s300_joint_2'))
    motors.append(robot.getMotor('j2s7s300_joint_3'))
    motors.append(robot.getMotor('j2s7s300_joint_4'))
    motors.append(robot.getMotor('j2s7s300_joint_5'))
    motors.append(robot.getMotor('j2s7s300_joint_6'))
    motors.append(robot.getMotor('j2s7s300_joint_7'))
    print("imported motors")
    return motors


def getAndInitAllSensorsJaco7(robot):
    sensors = []
    sensors.append(robot.getPositionSensor('j2s7s300_joint_1_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_2_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_3_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_4_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_5_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_6_sensor'))
    sensors.append(robot.getPositionSensor('j2s7s300_joint_7_sensor'))
   
   # now we must specify how often do sensors get read in miliseconds
   # i've put 10ms since this seems like it should work smoothly,
   # but ofc you should play with this value later on
    for sensor in sensors:
        sensor.enable(10)

    print("imported and inited sensors")
    return sensors

