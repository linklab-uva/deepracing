import py_f1_interface
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("steer", type=float)
    parser.add_argument("throttle", type=float)
    parser.add_argument("brake", type=float)
    args = parser.parse_args()
    controller = py_f1_interface.F1Interface(1)
    controller.setControl(args.steer,args.throttle,args.brake)
    
if __name__ == '__main__':
    main()