import time
import uinput
events = (uinput.BTN_JOYSTICK, uinput.ABS_Y + (0, 255, 0, 0))
device = uinput.Device(events)
time.sleep(2.5)
print("lez go")
device.emit(uinput.ABS_Y, 128, syn=False)
device.emit(uinput.BTN_X, 1, syn=False)
time.sleep(0.05)
device.emit(uinput.BTN_X, 0, syn=False)
time.sleep(0.05)
device.emit(uinput.BTN_X, 1, syn=False)
time.sleep(0.05)
device.emit(uinput.BTN_X, 0, syn=False)
for i in range(128):
    device.emit(uinput.ABS_Y, i)
    time.sleep(0.1)