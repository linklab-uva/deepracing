import py_vjoy
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--value", type=int, required=True)
    args = parser.parse_args()
    vjoy_angle = args.value
    vj = py_vjoy.vJoy()
    vj.capture(1) #1 is the device ID
    vj.reset()
    js = py_vjoy.Joystick()
    js.setAxisXRot(int(round(vjoy_angle))) 
    js.setAxisYRot(int(round(vjoy_angle))) 
    vj.update(js)
    
if __name__ == '__main__':
    main()