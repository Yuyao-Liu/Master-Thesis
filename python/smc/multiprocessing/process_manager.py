from multiprocessing import Process, Queue, Lock, shared_memory
import numpy as np
import pickle
import typing
from functools import partial
import copy


class ProcessManager:
    """
    ProcessManager
    --------------
    A way to do processing in a thread (process because GIL) different
    from the main one which is reserved
    for ControlLoopManager.
    The primary goal is to process visual input
    from the camera without slowing down control.
    TODO: once finished, map real-time-plotting and
    visualization with this (it already exists, just not in a manager).
    What this should do is hide the code for starting a process,
    and to enforce the communication defined by the user.
    To make life simpler, all communication is done with Queues.
    There are two Queues - one for commands,
    and the other for receiving the process' output.
    Do note that this is obviously not the silver bullet for every
    possible inter-process communication scenario,
    but the aim of this library is to be as simple as possible,
    not as most performant as possible.
    NOTE: the maximum number of things in the command queue is arbitrarily
        set to 5. if you need to get the result immediately,
        calculate whatever you need in main, not in a side process.
        this is meant to be used for operations that take a long time
        and aren't critical, like reading for a camera.
    """

    # NOTE: theoretically we could pass existing queues so that we don't
    # need to create new ones, but i won't optimize in advance
    def __init__(
        self, args, side_function, init_command, comm_direction, init_value=None
    ):
        self.args = args
        self.comm_direction = comm_direction

        # send command to slave process
        if comm_direction == 0:
            self.command_queue = Queue()
            self.side_process = Process(
                target=side_function,
                args=(
                    args,
                    init_command,
                    self.command_queue,
                ),
            )
        # get data from slave process
        if comm_direction == 1:
            self.data_queue = Queue()
            self.side_process = Process(
                target=side_function,
                args=(
                    args,
                    init_command,
                    self.data_queue,
                ),
            )
        # share data in both directions via shared memory with 2 buffers
        # - one buffer for master to slave
        # - one buffer for slave to master
        if comm_direction == 2:
            self.command_queue = Queue()
            self.data_queue = Queue()
            self.side_process = Process(
                target=side_function,
                args=(
                    args,
                    init_command,
                    self.command_queue,
                    self.data_queue,
                ),
            )
        # shared memory both ways
        # one lock because i'm lazy
        # but also, it's just copy-pasting
        # we're in python, and NOW do we get picky with copy-pasting???????
        if comm_direction == 3:
            # "sending" the command via shared memory
            # TODO: the name should be random and send over as function argument
            shm_name = "command"
            # NOTE: if we didn't close properly it will just linger on.
            # since there is no exist_ok argument for SharedMemory, we catch the error on the fly here
            try:
                self.shm_cmd = shared_memory.SharedMemory(
                    shm_name, create=True, size=init_command.nbytes
                )
            except FileExistsError:
                self.shm_cmd = shared_memory.SharedMemory(
                    shm_name, create=False, size=init_command.nbytes
                )

            self.shared_command = np.ndarray(
                init_command.shape, dtype=init_command.dtype, buffer=self.shm_cmd.buf
            )
            self.shared_command[:] = init_command[:]
            # same lock for both
            self.shared_command_lock = Lock()
            # getting data via different shared memory
            shm_data_name = "data"
            # size is chosen arbitrarily but 10k should be more than enough for anything really
            try:
                self.shm_data = shared_memory.SharedMemory(
                    shm_data_name, create=True, size=10000
                )
            except FileExistsError:
                self.shm_data = shared_memory.SharedMemory(
                    shm_data_name, create=False, size=10000
                )
            # initialize empty
            p = pickle.dumps(None)
            self.shm_data.buf[: len(p)] = p
            # the process has to create its shared memory
            self.side_process = Process(
                target=side_function,
                args=(
                    args,
                    init_command,
                    shm_name,
                    self.shared_command_lock,
                    self.shm_data,
                ),
            )
        # networking client (server can use comm_direction 0)
        if comm_direction == 4:
            from smc.multiprocessing.networking.util import DictPb2EncoderDecoder

            self.encoder_decoder = DictPb2EncoderDecoder()
            self.msg_code = self.encoder_decoder.dictToMsgCode(init_command)
            # TODO: the name should be random and send over as function argument
            shm_name = "client_socket" + str(np.random.randint(0, 1000))
            # NOTE: size is max size of the recv buffer too,
            # and the everything blows up if you manage to fill it atm
            self.shm_msg = shared_memory.SharedMemory(shm_name, create=True, size=1024)
            # need to initialize shared memory with init value
            # NOTE: EVIL STUFF SO PICKLING ,READ NOTES IN networking/client.py
            # init_val_as_msg = self.encoder_decoder.dictToSerializedPb2Msg(init_value)
            # self.shm_msg.buf[:len(init_val_as_msg)] = init_val_as_msg
            pickled_init_value = pickle.dumps(init_value)
            self.shm_msg.buf[: len(pickled_init_value)] = pickled_init_value
            self.lock = Lock()
            self.side_process = Process(
                target=side_function, args=(args, init_command, shm_name, self.lock)
            )
        if type(side_function) == partial:
            self.side_process.name = side_function.func.__name__
        else:
            self.side_process.name = side_function.__name__ + "_process"
        self.latest_data = init_value

        self.side_process.start()
        if self.args.debug_prints:
            print(f"PROCESS_MANAGER: i am starting {self.side_process.name}")

    # TODO: enforce that
    # the key should be a string containing the command,
    # and the value should be the data associated with the command,
    # just to have some consistency
    def sendCommand(self, command: typing.Union[dict, np.ndarray]) -> None:
        """
        sendCommand
        ------------
        assumes you're calling from controlLoop and that
        you want a non-blocking call.
        the maximum number of things in the command queue is arbitrarily
        set to 5. if you need to get the result immediately,
        calculate whatever you need in main, not in a side process.

        if comm_direction == 3:
        sendCommandViaSharedMemory
        ---------------------------
        instead of having a queue for the commands, have a shared memory variable.
        this makes sense if you want to send the latest command only,
        instead of stacking up old commands in a queue.
        the locking and unlocking of the shared memory happens here
        and you don't have to think about it in the control loop nor
        do you need to pollute half of robotmanager or whatever else
        to deal with this.
        """
        if self.comm_direction != 3:
            if self.command_queue.qsize() < 5:
                self.command_queue.put_nowait(command)

        if self.comm_direction == 3:
            assert type(command) == np.ndarray
            assert command.shape == self.shared_command.shape
            self.shared_command_lock.acquire()
            self.shared_command[:] = command[:]
            self.shared_command_lock.release()

    def getData(self) -> dict[str, typing.Any]:
        if self.comm_direction < 3:
            if not self.data_queue.empty():
                self.latest_data = self.data_queue.get_nowait()
        if self.comm_direction == 3:
            self.shared_command_lock.acquire()
            # here we should only copy, release the lock, then deserialize
            self.latest_data = pickle.loads(self.shm_data.buf)
            self.shared_command_lock.release()
        if self.comm_direction == 4:
            self.lock.acquire()
            # data_copy = copy.deepcopy(self.shm_msg.buf)
            # REFUSES TO WORK IF YOU DON'T PRE-CROP HERE!!!
            # MAKES ABSOLUTELY NO SENSE!!! READ MORE IN smc/networking/client.py
            # so we're decoding there, pickling, and now unpickling.
            # yes, it's incredibly stupid
            # new_data = self.encoder_decoder.serializedPb2MsgToDict(self.shm_msg.buf, self.msg_code)
            new_data = pickle.loads(self.shm_msg.buf)
            self.lock.release()
            if len(new_data) > 0:
                self.latest_data = new_data
            # print("new_data", new_data)
            # print("self.latest_data", self.latest_data)
            # self.latest_data = self.encoder_decoder.serializedPb2MsgToDict(data_copy, self.msg_code)
        return copy.deepcopy(self.latest_data)

    def terminateProcess(self) -> None:
        if self.comm_direction == 3:
            self.shm_cmd.close()
            self.shm_cmd.unlink()
        if (self.comm_direction != 3) and (self.comm_direction != 1):
            if self.args.debug_prints:
                print(
                    f"i am putting befree in {self.side_process.name}'s command queue to stop it"
                )
            self.command_queue.put_nowait("befree")
        try:
            self.side_process.terminate()
            if self.args.debug_prints:
                print(f"terminated {self.side_process.name}")
        except AttributeError:
            if self.args.debug_prints:
                print(f"{self.side_process.name} is dead already")
