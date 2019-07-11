import socketserver

class ImageTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        sizearr = self.request.recv(4)
        size = int.from_bytes(sizearr,byteorder='little')
        print(size)
        messagedata = self.request.recv(size).strip()
        ## just send back the same data, but upper-cased
        print(messagedata)
       # self.request.sendall(messagedata.upper())

if __name__ == "__main__":
    HOST, PORT = "localhost", 5005

    with socketserver.TCPServer((HOST, PORT), ImageTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()