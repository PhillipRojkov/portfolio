"""Serial communication for robot arms"""

import time
import struct
import serial

import numpy as np


class ArmSerial:
    """Encapsulates serial communication for robot arms"""
    # Configuration
    SYNC_WORD_1 = 0xAABBCCDD
    SYNC_WORD_2 = 0xBBAACCDD
    LEN_1 = 8
    LEN_2 = 8
    ACK_WORD = 0xDDCCBBAA
    SYNC_BYTES_1 = struct.pack('<I', SYNC_WORD_1)
    SYNC_BYTES_2 = struct.pack('<I', SYNC_WORD_2)
    ACK_BYTES = struct.pack('<I', ACK_WORD)

    def __init__(self, port='/dev/cu.usbmodem11401',
                 baudrate=115200,
                 arr_length=15,
                 ack_timeout=1.5,
                 max_retries=3):

        self.port = port
        self.baudrate = baudrate
        self.arr_length = arr_length
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries

        self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
        time.sleep(1)  # Arduino reset

    def poll_debug_output(self):
        """poll serial port for debug output from arudino"""
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    print("[ARDUINO]\r", line)
            except Exception:
                break

    def wait_for_ack(self):
        """Wait for ACK from arduino with timeout"""
        start = time.time()
        buffer = b''

        while time.time() - start < self.ack_timeout:
            # poll_debug_output(ser)

            byte = self.ser.read(1)
            if not byte:
                time.sleep(0.001)
                continue

            buffer += byte
            if len(buffer) > 4:
                buffer = buffer[-4:]
            if buffer == self.ACK_BYTES:
                return True
        return False

    def send_packet(self, data_array):
        """Send data_array packet of arr_length float list"""
        # Pack sync + float array
        # TODO Un hard-code the lengths
        packet = struct.pack('<I', self.SYNC_WORD_1)
        packet += struct.pack('<' + 'f' * self.LEN_1, *data_array[:self.LEN_1])

        success = False
        for attempt in range(1, self.max_retries + 1):
            self.ser.write(packet)
            print(f"Packet 1 sent (attempt {attempt})\r")

            if self.wait_for_ack():
                print("ACK received ✅\r")
                success = True
                break
            print("ACK timeout ❌\r")

        if not success:
            print("Failed to receive ACK after retries\r")

    def send_ik(self, ik1, grip1_closed, move_time=2.0, verbose=False):
        grip_closed_angle = 27
        # Send data
        data_array = ik1[1:].tolist()
        if grip1_closed:
            data_array.append(grip_closed_angle*np.pi/180)
        else:
            data_array.append(10*np.pi/180)

        data_array.append(float(move_time))

        if verbose:
            print(f'angles {data_array}\r')

        self.send_packet(data_array)
