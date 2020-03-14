import py_f1_interface
import argparse
parser = argparse.ArgumentParser(description="Set steering ratio")
parser.add_argument("angle", type=float, help="Steering value [-1,1]")
parser.add_argument("throttle", type=float, help="Steering value [-1,1]")
parser.add_argument("brake", type=float, help="Steering value [-1,1]")
args = parser.parse_args()
controller = py_f1_interface.F1Interface(1)
controller.setControl(args.angle, args.throttle, args.brake)