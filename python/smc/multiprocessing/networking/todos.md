1. rename receiver and sender to subscriber and publisher
2. switch over to UDP - just drop a packed if deserialization fails + log it ideally
3. add a checksum to see if packet is correct, for example:
import zlib
import sensor_pb2  # Generated from your .proto file

def serialize_with_checksum(sensor_data):
    # Clear checksum before computing
    sensor_data.checksum = 0

    # Serialize message (excluding checksum)
    serialized_data = sensor_data.SerializeToString()

    # Compute CRC32 checksum
    checksum = zlib.crc32(serialized_data) & 0xFFFFFFFF  # Ensure unsigned 32-bit

    # Set checksum field
    sensor_data.checksum = checksum

    # Serialize again with checksum
    return sensor_data.SerializeToString()

and similar to deserialize
def deserialize_with_checksum(data):
    received_data = sensor_pb2.SensorData()
    received_data.ParseFromString(data)

    # Extract the received checksum
    received_checksum = received_data.checksum

    # Recompute the checksum (excluding checksum field)
    received_data.checksum = 0
    recalculated_checksum = zlib.crc32(received_data.SerializeToString()) & 0xFFFFFFFF

    # Validate checksum
    if received_checksum != recalculated_checksum:
        raise ValueError("Checksum mismatch! Corrupted data received.")

    return received_data  # Valid message

