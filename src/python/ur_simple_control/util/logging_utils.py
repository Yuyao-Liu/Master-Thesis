import pickle
import numpy as np
import os
import subprocess
import re
from ur_simple_control.visualize.visualize import plotFromDict



class LogManager:
    """
    LogManager
    ----------
    The simplest possible way to combine logs of different 
    control loops - store them separately.
    Comes with some functions to clean and combine logs
    of different control loops (TODO).
    - input: log_dicts from ControlLoopManager
    - output: saves this whole class as pickle -
              data and arguments included
    """
    def __init__(self, args):
        if args is None:
            return
        self.args = args
        self.are_logs_vectorized_flag = False
        self.loop_logs = {}
        self.loop_number = 0
        # name the run
        self.run_name = 'latest_run'
        if self.args.run_name != 'latest_run':
            self.run_name = self.args.run_name
        # assign save directory
        if args.save_dir != "./data":
            if os.path.exists(self.args.save_dir):
                self.save_dir = self.args.save_dir 
        else:
            if os.path.exists("./data"):
                self.save_dir = "./data"
            else:
                os.makedirs('/tmp/data', exist_ok=True)
                self.save_dir = '/tmp/data'
            
        # if indexing (same run, multiple times, want to save all) 
        # update the name with the free index
        if args.index_runs:
            index = self.findLatestIndex()
            self.run_name = self.run_name + "_" + str(index) + ".pickle"

        self.save_file_path = os.path.join(self.save_dir, self.run_name)


    def storeControlLoopRun(self, log_dict, loop_name, final_iteration):
        loop_name = str(self.loop_number) + '_' + loop_name
        self.loop_number += 1
        self.loop_logs[loop_name] = log_dict

    def vectorizeLog(self):
        for control_loop_name in self.loop_logs:
            for key in self.loop_logs[control_loop_name]:
                self.loop_logs[control_loop_name][key] = np.array(self.loop_logs[control_loop_name][key])
        self.are_logs_vectorized_flag = True

    def saveLog(self, cleanUpRun=False):
        """
        saveLog
        -------
        transforms the logs obtained from control loops
        into numpy arrays and pickles the whole of LogManager
        (including the data and the args).
        Uses default pickling.
        """
        # finally transfer to numpy (after nothing is running anymore)
        if not self.are_logs_vectorized_flag:
            self.vectorizeLog()
        if cleanUpRun:
            self.cleanUpRun()
        print(f"data is ready, logmanager will now save your logs to \
                {self.save_file_path}")
        log_file = open(self.save_file_path, 'wb')
        pickle.dump(self.__dict__, log_file)
        log_file.close()

    def loadLog(self, save_file_path):
        """
        loadLog
        -------
        unpickles a log, which is the whole of LogManager
        (including the data and the args).
        Uses default (un)pickling.
        """
        if os.path.exists(save_file_path):
            log_file = open(save_file_path, 'rb')
            tmp_dict = pickle.load(log_file)
            log_file.close()
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)
        else:
            print("you didn't give me a correct save_file_path! exiting")
            exit()

    def plotAllControlLoops(self):
        if not self.are_logs_vectorized_flag:
            self.vectorizeLog() 
            self.are_logs_vectorized_flag = True

        for control_loop_name in self.loop_logs:
            plotFromDict(self.loop_logs[control_loop_name], len(self.loop_logs[control_loop_name]['qs']), self.args, title=control_loop_name)


    def findLatestIndex(self):
        """
        findLatestIndex
        ---------------
        reads save_dir, searches for run_name,
        finds the highest index within the file whose names match run_name.
        NOTE: better to not have other files in the data dir,
        this isn't written to work in every circumstances,
        it assumes a directory with Simple Manipulator Control log files only
        """
        child = subprocess.Popen(['ls', self.save_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        files_in_datadir = child.stdout.read().decode('utf-8').split("\n")
        regex_name = re.compile(self.run_name + ".*") 
        regex_index = re.compile("[0-9]+") 
        highest_index = -1
        for file_name in files_in_datadir:
            rez_name = regex_name.search(file_name)
            if rez_name != None:
                rez_index = regex_index.findall(file_name)
                if len(rez_index) > 0:
                    this_file_name_index = int(rez_index[-1])
                    if this_file_name_index > highest_index:
                        highest_index = this_file_name_index

        index = highest_index + 1
        return index
