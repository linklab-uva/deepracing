import socketserver
import Image_pb2
import ChannelOrder_pb2
import google.protobuf.json_format
import numpy as np
import PIL
import cv2
def channelOrderToNumChannels(channel_order : int):
    if channel_order==ChannelOrder_pb2.ChannelOrder.GRAYSCALE:
        return 1
    elif channel_order==ChannelOrder_pb2.ChannelOrder.BGRA:
        return 2
    else:
        return 3
class ImageTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    #def __init__(self, request, client_address, server):
    #    self.protocol="ascii"
    #    return super().__init__(request, client_address, server)
    def handle(self):
        try:# self.request is the TCP socket connected to the client
            sizearr = self.request.recv(4)
            size = int.from_bytes(sizearr,byteorder='little', signed=False)
            print("Received %u bytes" %(size))
            messagedata = self.request.recv(size)
            #print(messagestring)
            
            #messagestring = messagedata.decode("ascii")
            #image_data = google.protobuf.json_format.Parse(messagestring, Image_pb2.Image())

            image_data_in = Image_pb2.Image()
            image_data_in.ParseFromString(messagedata)#.copy())


            #print(image_data.image_data)
            print(image_data_in.rows)
            print(image_data_in.cols)
            print(image_data_in.channel_order)
            im = np.frombuffer(image_data_in.image_data, dtype=np.uint8).reshape((image_data_in.rows, image_data_in.cols, channelOrderToNumChannels(image_data_in.channel_order)))
           # print(im)

            im_reply_pb = Image_pb2.Image()
            im_out = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)

            im_reply_pb.rows = im_out.shape[0]
            im_reply_pb.cols = im_out.shape[1]
            im_reply_pb.channel_order = ChannelOrder_pb2.ChannelOrder.GRAYSCALE
            im_reply_pb.image_data = im_out.flatten().tobytes()
            replysize = im_reply_pb.ByteSize()
            print( "Responding with %d bytes" % (replysize) )
            replysizearr =  replysize.to_bytes(4, byteorder='little', signed=False)
            print( replysizearr )
            replymessagedata =  im_reply_pb.SerializeToString()

            self.request.send(replysizearr + replymessagedata)
        except:
            self.request.send(str.encode("NO","ascii"))

if __name__ == "__main__":
    HOST, PORT = "localhost", 5005

    with socketserver.TCPServer((HOST, PORT), ImageTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()