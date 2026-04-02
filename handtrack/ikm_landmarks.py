import matplotlib.pyplot
import numpy as np
import cv2

import time

from curses import wrapper
import curses

import argparse

import arm_serial
import arm_controller

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import hand_landmarks
import calibration_io

do_serial = False
do_visual = False

left_arm_offset = arm_controller.LEFT_OFFSET_Y # m
controller_right = arm_controller.Arm("right_arm")
controller_left = arm_controller.Arm("left_arm", origin_offset=[0, left_arm_offset, 0])

def main(stdscr):
    angle_step = 10 * np.pi / 180  # rad
    translation_step = 5 / 100  # m
    speed = 0.5 # seconds per movement
    big_move_speed = 1.5

    smoothed_hand_world = None
    alpha = 0.2  # 0 < alpha <= 1 (lower = smoother, more lag)

    angle_step_change_factor = 0.2
    translation_step_change_factor = 0.2

    parser = argparse.ArgumentParser("Inverse Kinematics Test")
    parser.add_argument("-s", "--serial", action='store_true', default=False)
    parser.add_argument("-v", "--visualize", action='store_true', default=True)
    parser.add_argument("-p", "--port", default="/dev/cu.usbmodem11401")
    parser.add_argument("-c", "--camera", default=0)
    parser.add_argument("-e", "--speed", default=1.0) # Speed factor for hand to arm
    args = parser.parse_args()
    do_serial = args.serial
    do_visual = args.visualize
    ser_port = args.port
    camera_index = args.camera
    hand_speed = float(args.speed)

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

    # hand, arm coords at start of tracking
    init_hand_coords = None
    init_arm_coords = None

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera_calibration = calibration_io.load_fisheye_params()

    handLandmarks = hand_landmarks.HandLandmarks()

    move_left = True
    move_right = True
    grip_left = True
    grip_right = True
    delta_time = 0.05 # loop sleep time
    prev_loop_time = 0
    while True:
        new_pose = True
        tracking_hand = False
        position_left = pose_left[:3]
        angles_left = pose_left[3:]
        position_right = pose_right[:3]
        angles_right = pose_right[3:]

        ret, frame_bgr = cap.read()
        frame_bgr = calibration_io.undistort_frame(frame_bgr, camera_calibration)

        big_move = False

        stdscr.clear()
        stdscr.refresh()
        print("| wasdqe orientation | arrow keys and +/- translation |\r")
        print("| l, r toggle left/right control                      |\r")
        print("| spacebar hand tracking                              |\r")
        print("| f grip             | t change speed                 |\r")
        print("| m manual mode      | numkeys saved pos              |\r")
        print("| i home             | <, > move less, move more      |\r")
        print("| x exit             | v pause to interact with plot  |\r")
        print(f'Left control: {move_left}, right control: {move_right}\r')
        print(f'position left: {position_left}, position right: {position_right}\r')
        print(f'angles left: {angles_left}, angles right: {angles_right}\r')
        print(f'position step: {translation_step:.3f} m, angle step: {(angle_step * 180 / np.pi):.3f} deg\r')
        print(f'move_time (s): {speed:.3f}\r')
        dt = time.time() - prev_loop_time
        print(f'loop period (s): {dt:.3f}\r')
        prev_loop_time = time.time()
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
            position_left[2] = 0.01
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.3
            position_right[1] = 0
            position_right[2] = 0.01
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
            big_move = True
        elif key == ord('2'):
            position_left[0] = 0.42
            position_left[1] = 0.34
            position_left[2] = 0.15
            angles_left[0] = 0
            angles_left[1] = 0
            angles_left[2] = 0
            position_right[0] = 0.43
            position_right[1] = 0.15
            position_right[2] = 0.15
            angles_right[0] = 0
            angles_right[1] = 0
            angles_right[2] = 0
            big_move = True
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
            big_move = True
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
            big_move = True
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
            big_move = True
        elif key == ord('x'):
            cap.release()
            cv2.destroyAllWindows()
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
            big_move = True
        elif key == ord('r'):
            move_right = not move_right
        elif key == ord('l'):
            move_left = not move_left
        elif key == ord(' '):
            tracking_hand = True

        if key == -1:
            new_pose = False

        # Clear out any other characters that have been buffered
        while stdscr.getch() != -1:
            pass

        hand_coords, hand_grip, results = handLandmarks.getCoords(frame_bgr, camera_calibration.k, grip_dist=0.05)
        # hand_coords, hand_grip, results = handLandmarks.getCoords(frame_bgr, None, grip_dist=0.06)
        # Convert to hand coords to wrist frame
        # Camera coords are +x right, +y down, +z out
        # Camera is tilted 60 degrees down about x axis
        if hand_coords is not None:
            R_align = np.array([
                [0,  0,  1],   # cam z → robot x
                [-1, 0,  0],   # cam x → -robot y
                [0, -1,  0]    # cam y → -robot z
            ])
            theta = -np.pi / 3
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]
            ])
            hand_world = R_align @ Rx @ hand_coords

            cv2.putText(frame_bgr, f"Camera: ({hand_coords[0]:.3f}, {hand_coords[1]:.3f}, {hand_coords[2]:.3f})", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_bgr, f"World: ({hand_world[0]:.3f}, {hand_world[1]:.3f}, {hand_world[2]:.3f})", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_bgr, f"Grip: {hand_grip}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if hand_grip else (0, 0, 255), 2)
            frame_bgr = handLandmarks.draw_landmarks_on_image(frame_bgr, results)
    
        if tracking_hand:
            if hand_coords is not None:
                # init_hand_coords being None indicates that this is the first iteration of tracking
                if init_hand_coords is None or init_arm_coords is None:
                    init_hand_coords =  hand_world
                    init_arm_coords = position_right.copy()
                    smoothed_hand_world = hand_world.copy()
                else:
                    smoothed_hand_world[:] = (
                        alpha * hand_world + (1 - alpha) * smoothed_hand_world
                    )

                position_right[:] = init_arm_coords + hand_speed * (
                    smoothed_hand_world - init_hand_coords
                )
                grip_right = hand_grip
        else:
            init_hand_coords = None
            init_arm_coords = None
            smoothed_hand_world = None

        # Run Controller
        # controller_left.update_from_pose(pose_left, grip_left)
        controller_right.update_from_pose(pose_right, grip_right)

        if do_visual:
            ax.clear()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            # controller_left.arm_kinematics.arm_chain.plot(controller_left.motor_angles, ax, show=False)
            controller_right.arm_kinematics.arm_chain.plot(controller_right.motor_angles, ax, show=False)
            fig.canvas.draw()
            ax.figure.canvas.flush_events()

        cv2.imshow("RunCam Live", frame_bgr)
        # Important: waitKey is required for window events on macOS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        if do_serial and new_pose:
            speed_ = speed
            if tracking_hand:
                speed_ = 0 # Indicates linear interpolation in firmware
            elif big_move:
                speed_ = big_move_speed
            ser.send_ik(controller_right.motor_angles, controller_right.gripper_closed, move_time=speed_)

        # time.sleep(delta_time)

wrapper(main)
