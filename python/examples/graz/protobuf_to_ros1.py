import socket
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
import time
import argparse
import message_specs_pb2
import rospy
from geometry_msgs.msg import WrenchStamped


def getArgs():
    parser = argparse.ArgumentParser()
    parser.description = (
        "get ros1 message, make it a protobuf message, send it via tcp socket"
    )
    parser.add_argument("--host", type=str, help="host ip address", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="host's port", default=6666)

    args = parser.parse_args()
    return args


def parse_message(buffer):
    """
    parse_message
    -------------
    here the message is what we got from recv(),
    and parsing it refers to finding serialized pb2 messages in it
    NOTE: we only keep the latest message because
          we're not going to do anything with the missed message.
          the assumption is that we only get the same kind of message from
          a sensor or something similar, not files or whatever else needs to be whole
    """
    pos, next_pos = 0, 0
    buffer_len = len(buffer)
    print("buffer_len", buffer_len)
    msg_in_bytes = b""
    len_size_offset = 0
    while True:
        next_pos, pos = _DecodeVarint32(buffer, pos)
        if pos + next_pos > buffer_len:
            return msg_in_bytes, pos - len_size_offset
        msg_in_bytes = _VarintBytes(pos + next_pos) + buffer[pos : pos + next_pos]
        len_size_offset = len(_VarintBytes(pos + next_pos))
        pos += next_pos
        if pos >= buffer_len:
            return msg_in_bytes, pos



# ros parts
#rospy.init_node("ur5e_to_ros1_mapper", anonymous=True)
#pub = rospy.Publisher("wrench_msg_mapper", WrenchStamped, queue_size=10)

args = getArgs()
host_addr = (args.host, args.port)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
buffer = b""
while True:
    try:
        s.connect(host_addr)
        break
    except ConnectionRefusedError:
        time.sleep(0.005)
print("NETWORKING CLIENT: connected to server")
s.settimeout(None)

#try:
while True:
    msg_raw = s.recv(1024)
    buffer += msg_raw
    print(msg_raw)
    print(len(msg_raw))
    if len(msg_raw) < 1:
        continue
    msg_in_bytes, pos = parse_message(buffer)
    buffer = buffer[pos:]

    next_pos, pos = 0, 0
    next_pos, pos = _DecodeVarint32(msg_in_bytes, pos)
    pb2_msg_in_bytes_cut = msg_in_bytes[pos : pos + next_pos]
    pb2_msg = message_specs_pb2.wrench()
    pb2_msg.ParseFromString(pb2_msg_in_bytes_cut)

    print(pb2_msg.wrench)
    # here you send construct and send the ros message
    #wrench_message = WrenchStamped()
    #wrench_message.wrench.force.x = pb2_msg.wrench[0]
    #wrench_message.wrench.force.y = pb2_msg.wrench[1]
    #wrench_message.wrench.force.z = pb2_msg.wrench[2]
    #wrench_message.wrench.torque.x = pb2_msg.wrench[3]
    #wrench_message.wrench.torque.y = pb2_msg.wrench[4]
    #wrench_message.wrench.torque.z = pb2_msg.wrench[5]
    #wrench_message.header.stamp = rospy.Time.now()
    #wrench_message.header.frame_id = 0
    #pub.publish(wrench_message)
    # time.sleep(0.002)

#except KeyboardInterrupt:
#    s.close()
#    print("NETWORKING_CLIENT: caught KeyboardInterrupt, i'm out")
