# TODO: make protobuf an optional import
#       then if it's not there use pickling as default
import smc.multiprocessing.networking.message_specs_pb2 as message_specs_pb2

from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
import numpy as np
import pinocchio as pin
import pickle

"""
to make the data API uniform across the library (dictionaries),
we need to have a lookup function between these dictionaries 
and messages send with protobuf.
i don't want to see protobuf construction in my control loop aight.

this is done as follows.
there is a map between a binary number and messages.
the binary number is of fixed length, and each bit represents
a boolean for whether a specific quantity is in the keys of the dictionary.
it is initialized to all zeros (false)
when a dictionary is inputed, the function goes over every key and sets
the corresponding value.
of course for every combination there is a pre-defined message in protobuf.
(well it won't be, that's a todo because it might be better
to have a big message with optional fields or something).
the output is a message with the message values put in.

is this the best way of dealing with enabling unifromness in the API?
is this the best way to construct protobuf messages?
i don't know, but it solves my most important problem - i don't
want to look at protobuf in the rest of the code
"""


class DictPb2EncoderDecoder:
    def __init__(self):
        # TODO: fill this in with more messages
        self.key_to_index = {
            "q": 0,
            "wrench": 1,
            "wrench_estimate": 2,
            "T_goal": 3,
            "v": 4,
        }
        # if you want to embed message codes into messages,
        # you'll need to use this
        # stupidest possible solution

        # TODO: implement own bidirectional dict for instead of this
        # self.index_to_key = {self.key_to_index[key]: key for key in self.key_to_index}

    def dictToMsgCode(self, dict_msg):
        """
        you give me a dict that you want to send,
        i give you it's code which internally corresponds
        to the pb2 message which the given dict should form
        """
        msg_code = 0b000
        for key in dict_msg:
            # 1 << x gets you a binary starting with 1 followed by x zeros
            # | is binary or
            # because i can't just flip a bit because python.
            # this certainly isn't the most efficient way to go about it,
            # but what can you do
            msg_code = msg_code | (1 << self.key_to_index[key])
        return msg_code

    def dictToSerializedPb2Msg(self, dict_msg):
        """
        dictToPb2Msg
        ------------
        takes dict, finds it's code and the corresponding pb2 message,
        fills in the values, serializes
        the message and prepends its length.

        NOTE: possible extension: prepend msg_code as well,
        that way you can send different messages
        you could also just have multiple sockets, all sending the same
        message. ask someone who actually knows network programming
        what makes more sense
        """
        msg_code = self.dictToMsgCode(dict_msg)
        if msg_code == 1:
            pb2_msg = message_specs_pb2.joint_angles()
            # if i could go from string to class atribute somehow
            # that'd be amazing
            pb2_msg.q.extend(dict_msg["q"])
        if msg_code == 2:
            pb2_msg = message_specs_pb2.wrench()
            # if i could go from string to class atribute somehow
            # that'd be amazing
            pb2_msg.wrench.extend(dict_msg["wrench"])
        if msg_code == 6:
            pb2_msg = message_specs_pb2.wrenches()
            # if i could go from string to class atribute somehow
            # that'd be amazing
            pb2_msg.wrench.extend(dict_msg["wrench"])
            pb2_msg.wrench_estimate.extend(dict_msg["wrench_estimate"])
        if msg_code == 24:
            pb2_msg = message_specs_pb2.T_goal()
            # if i could go from string to class atribute somehow
            # that'd be amazing
            pb2_msg.position.extend(dict_msg["T_goal"].translation)
            pb2_msg.rotation.extend(
                pin.Quaternion(dict_msg["T_goal"].rotation).coeffs()
            )
            pb2_msg.velocity.extend(dict_msg["v"])

        # NOTE: possible extension:
        #   prepend the message code as well so that i can send different
        #   messages over the same socket.
        #   might come in handy and it comes for the price of one int per message,
        #   which sounds like a good deal to me
        #   msg_serialized = _VarintBytes(msg_code) + pb2_msg.SerializeToString()
        #    --> not bothering with it until necessary
        msg_length = pb2_msg.ByteSize()
        msg_serialized = pb2_msg.SerializeToString()
        # NOTE: protobuf is not self-delimiting so we have to prepend
        # the length of each message to know how to parse multiple
        # messages if they get buffered at the client (likely occurance)
        msg = _VarintBytes(msg_length) + msg_serialized
        return msg

    def serializedPb2MsgToDict(self, pb2_msg_in_bytes, msg_code):
        """
        pb2MsgToDict
        ------------
        input is pb2 msg in bytes prepended with msg length and
        msg_code, in that order, both are integers

        atm assumes only one message type will be received,
        and that you have to pass as an argument (it's the one
        you get when you call dictToMsgCode).

        alternatively, as an extension,
        the msg_code is embeded in the message and then you can send
        different messages over the same socket.
        """
        dict_msg = {}
        # who knows what goes on in the rest of the shared memory
        # protobuf is not self delimiting
        # so the first value always have to the length,
        # and we only read pass just the actual message to ParseFromString()
        next_pos, pos = 0, 0
        next_pos, pos = _DecodeVarint32(pb2_msg_in_bytes, pos)
        # print("did decode the int")
        # print("pos", pos, "next_pos", next_pos)
        pb2_msg_in_bytes_cut = pb2_msg_in_bytes[pos : pos + next_pos]
        if msg_code == 1:
            pb2_msg = message_specs_pb2.joint_angles()
            # pb2_msg.ParseFromString(pb2_msg_in_bytes[pos : pos + next_pos])
            # print("msg", pb2_msg)
            pb2_msg.ParseFromString(pb2_msg_in_bytes_cut)
            # print("msg", pb2_msg)
            # if i could go from string to class atribute somehow
            # that'd be amazing
            # TODO: see if that's possible
            dict_msg["q"] = np.array(pb2_msg.q)
        if msg_code == 2:
            pb2_msg = message_specs_pb2.wrench()
            pb2_msg.ParseFromString(pb2_msg_in_bytes_cut)
            # if i could go from string to class atribute somehow
            # that'd be amazing
            dict_msg["wrench"] = np.array(pb2_msg.wrench)
        if msg_code == 6:
            pb2_msg = message_specs_pb2.wrenches()
            pb2_msg.ParseFromString(pb2_msg_in_bytes_cut)
            # if i could go from string to class atribute somehow
            # that'd be amazing
            # TODO: see if that's possible
            dict_msg["wrench"] = np.array(pb2_msg.wrench)
            dict_msg["wrench_estimate"] = np.array(pb2_msg.wrench_estimate)
        if msg_code == 24:
            pb2_msg = message_specs_pb2.T_goal()
            pb2_msg.ParseFromString(pb2_msg_in_bytes_cut)
            # if i could go from string to class atribute somehow
            # that'd be amazing
            dict_msg["T_goal"] = pin.XYZQUATToSE3(
                list(pb2_msg.position) + list(pb2_msg.rotation)
            )
            dict_msg["v"] = np.array(pb2_msg.velocity)

        return dict_msg


class DictPickleWithHeaderEncoderDecoder:
    def __init__(self):
        self.HEADERSIZE = 10
        self.buffer = b""

    def dictToSerializedMsg(self, dict_msg: dict):
        msg = pickle.dumps(dict_msg)
        return bytes(f"{len(msg)}:<{self.HEADERSIZE}", "utf-8") + msg

    def what(self):
        pass
