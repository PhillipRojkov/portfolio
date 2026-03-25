import matplotlib.pyplot
import numpy as np

import time

from curses import wrapper
import curses

import argparse

import arm_serial
import arm_controller

do_serial = False
do_visual = False

left_arm_offset = arm_controller.LEFT_OFFSET_Y # m
controller_right = arm_controller.Arm("right_arm")
controller_left = arm_controller.Arm("left_arm", origin_offset=[0, left_arm_offset, 0])

def main(stdscr):
    angle_step = 10 * np.pi / 180  # rad
    translation_step = 5 / 100  # m
    speed = 2.0 # seconds per movement

    angle_step_change_factor = 0.2
    translation_step_change_factor = 0.2

    parser = argparse.ArgumentParser("Inverse Kinematics Test")
    parser.add_argument("-s", "--serial", action='store_true', default=False)
    parser.add_argument("-v", "--visualize", action='store_true', default=True)
    parser.add_argument("-p", "--port", default="/dev/cu.usbmodem11401")
    args = parser.parse_args()
    do_serial = args.serial
    do_visual = args.visualize
    ser_port = args.port

    stdscr.nodelay(True)
    stdscr.clear()

    if do_visual:
        matplotlib.pyplot.ion()
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

    if do_serial:
        ser = arm_serial.ArmSerial(port=ser_port)

    pose_left = controller_left.pose
    pose_right = controller_right.pose

    move_left = True
    move_right = True
    grip_left = True
    grip_right = True
    delta_time = 0.1 # loop period
    while True:
        new_pose = True
        position_left = pose_left[:3]
        angles_left = pose_left[3:]
        position_right = pose_right[:3]
        angles_right = pose_right[3:]

        stdscr.clear()
        stdscr.refresh()
        print("| wasdqe orientation | arrow keys and +/- translation |\r")
        print("| l, r toggle left/right control                      |\r")
        print("| f grip             | t change speed                 |\r")
        print("| m manual mode      | numkeys saved pos              |\r")
        print("| i home             | <, > move less, move more      |\r")
        print("| x exit             | v pause to interact with plot  |\r")
        print(f'Left control: {move_left}, right control: {move_right}\r')
        print(f'position left: {position_left}, position right: {position_right}\r')
        print(f'angles left: {angles_left}, angles right: {angles_right}\r')
        print(f'position step: {translation_step:.3f} m, angle step: {(angle_step * 180 / np.pi):.3f} deg\r')
        print(f'speed: {speed:.3f}\r')
        key = stdscr.getch()

        if key == curses.KEY_UP:
            if move_left:
                position_left[0] += translation_step
            if move_right:
                position_right[0] += translation_step
        elif key == curses.KEY_DOWN:
            if move_left:
                position_left[0] -= translation_step
            if move_right:
                position_right[0] -= translation_step
        elif key == curses.KEY_LEFT:
            if move_left:
                position_left[1] += translation_step
            if move_right:
                position_right[1] += translation_step
        elif key == curses.KEY_RIGHT:
            if move_left:
                position_left[1] -= translation_step
            if move_right:
                position_right[1] -= translation_step
        elif key == ord('='):
            if move_left:
                position_left[2] += translation_step
            if move_right:
                position_right[2] += translation_step
        elif key == ord('-'):
            if move_left:
                position_left[2] -= translation_step
            if move_right:
                position_right[2] -= translation_step
        elif key == ord('q'):
            if move_left:
                angles_left[0] -= angle_step
            if move_right:
                angles_right[0] -= angle_step
        elif key == ord('e'):
            if move_left:
                angles_left[0] += angle_step
            if move_right:
                angles_right[0] += angle_step
        elif key == ord('w'):
            if move_left:
                angles_left[1] -= angle_step
            if move_right:
                angles_right[1] -= angle_step
        elif key == ord('s'):
            if move_left:
                angles_left[1] += angle_step
            if move_right:
                angles_right[1] += angle_step
        elif key == ord('a'):
            if move_left:
                angles_left[2] += angle_step
            if move_right:
                angles_right[2] += angle_step
        elif key == ord('d'):
            if move_left:
                angles_left[2] -= angle_step
            if move_right:
                angles_right[2] -= angle_step
        elif key == ord('f'):
            if move_left:
                grip_left = not grip_left
            if move_right:
                grip_right = not grip_right
        elif key == ord('m'):
            stdscr.nodelay(False)
            print("Enter arm number (l, r)")
            arm = stdscr.getch()
            if arm == ord('r'): # right
                print("Setting right arm\r")
                print("Enter x coordinate (m)\r")
                position_right[0] = float(stdscr.getstr())
                print("Enter y coordinate (m)\r")
                position_right[1] = float(stdscr.getstr())
                print("Enter z coordinate (m)\r")
                position_right[2] = float(stdscr.getstr())
            elif arm == ord('l'): # left
                print("Setting left arm\r")
                print("Enter x coordinate (m)\r")
                position_left[0] = float(stdscr.getstr())
                print("Enter y coordinate (m)\r")
                position_left[1] = float(stdscr.getstr())
                print("Enter z coordinate (m)\r")
                position_left[2] = float(stdscr.getstr())
            else:
                print("Invalid arm!\r")
            stdscr.nodelay(True)
        elif key == ord('t'):
            stdscr.nodelay(False)
            print(f"Enter speed (seconds per movement)\r")
            speed = float(stdscr.getstr())
            stdscr.nodelay(True)
        elif key == ord(','): # <
            translation_step *= 1 - translation_step_change_factor
            angle_step *= 1 - angle_step_change_factor
            print(f'position step: {translation_step} m')
            print(f'angle step: {angle_step} rad')
        elif key == ord('.'): # >
            translation_step *= 1 + translation_step_change_factor
            angle_step *= 1 + angle_step_change_factor
        elif key == ord('1'):
            position_left[0] = 0.3 + 0.185
            position_left[1] = 0.185
            position_left[2] = 0.085
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.3
            position_right[1] = 0
            position_right[2] = 0.085
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
        elif key == ord('2'):
            position_left[0] = 0.42
            position_left[1] = 0.34
            position_left[2] = 0.1
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.43
            position_right[1] = 0.15
            position_right[2] = 0.1
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
        elif key == ord('3'):
            position_left[0] = 0.8
            position_left[1] = 0.4
            position_left[2] = 0.15
            angles_left[0] = 0
            angles_left[1] = 0.65
            angles_left[2] = 0
            position_right[0] = 0.8
            position_right[1] = 0
            position_right[2] = 0.15
            angles_right[0] = 0
            angles_right[1] = 0.65
            angles_right[2] = 0
        elif key == ord('4'):
            position_left[0] = 0.2
            position_left[1] = 0.3
            position_left[2] = 0.7
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.2
            position_right[1] = 0.1
            position_right[2] = 0.7
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
        elif key == ord('5'):
            position_left[0] = 0.2
            position_left[1] = 0.4
            position_left[2] = 0.75
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.2
            position_right[1] = 0
            position_right[2] = 0.75
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
        elif key == ord('x'):
            exit(0)
        elif key == ord('v'):
            matplotlib.pyplot.pause(10)
        elif key == ord('i'):
            position_left[0] = 0.3
            position_left[1] = 0 + left_arm_offset
            position_left[2] = 0.2
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.3
            position_right[1] = 0
            position_right[2] = 0.2
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
        elif key == ord('r'):
            move_right = not move_right
        elif key == ord('l'):
            move_left = not move_left

        if key == -1:
            new_pose = False

        # Clear out any other characters that have been buffered
        while stdscr.getch() != -1:
            pass

        # Run Controller
        controller_left.update_from_pose(pose_left, grip_left)
        controller_right.update_from_pose(pose_right, grip_right)

        if do_visual:
            ax.clear()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            controller_left.arm_kinematics.arm_chain.plot(controller_left.motor_angles, ax, show=False)
            controller_right.arm_kinematics.arm_chain.plot(controller_right.motor_angles, ax, show=False)
            fig.canvas.draw()
            ax.figure.canvas.flush_events()

        if do_serial and new_pose:
            ser.send_ik(controller_right.motor_angles, controller_left.motor_angles, controller_right.gripper_closed, controller_left.gripper_closed, movement_time=speed)

        time.sleep(delta_time)

wrapper(main)
