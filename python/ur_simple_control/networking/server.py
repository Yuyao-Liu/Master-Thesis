import socket
from google.protobuf.internal.encoder import _VarintBytes
from ur_simple_control.networking.util import DictPb2EncoderDecoder


def server(args, init_command, queue):
    """
    server
    -------
    listens for a connection, then sends messages to the singular accepted client

    ex. host = "127.0.0.1"
    ex. host_port = 7777

    use comm_direction = 0 in processmanager for this
    """
    encoder_decoder = DictPb2EncoderDecoder()
    host_addr = (args.host, args.port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(host_addr)
    if args.debug_prints:
        print("NETWORKING_SERVER: server listening on", host_addr)
    s.listen()
    comm_socket, comm_addr = s.accept()
    # we're only accepting a single connection
    s.close()
    if args.debug_prints:
        print("NETWORKING_SERVER: accepted a client", comm_addr)
    try:
        while True:
            # the construction of the message should happen in processmanager
            msg_dict = queue.get()
            if msg_dict == "befree":
                if args.debug_prints:
                    print("NETWORKING_SERVER: got befree, networking server out")
                break
            comm_socket.send(encoder_decoder.dictToSerializedPb2Msg(msg_dict))
    except KeyboardInterrupt:
        if args.debug_prints:
            print("NETWORKING_SERVER: caugth KeyboardInterrupt, networking server out")
    comm_socket.close()
