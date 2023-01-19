###
###  connect4.parallel.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of abstract objects for running data generation methods in parallel subprocesses, managed by parallel worker and monitor threads.
"""

import sys, time
from __future__ import annotations

from threading import Lock, Thread

from multiprocess import Process, Queue


##  Global container of all custom threads created by this module
all_threads = []
        


###=================================###
###   BaseThread class definition   ###
###=================================###
        
class BaseThread(Thread) :
    
    def __init__(self):
        """
        Base class for creating custom threads. 
           > kill method sets an internal flag that orders thread to stop at the next opportunity
           > info method provides safe printing to stdout
           > on __init__, thread is automatically added to the global list of all threads created by this module
        """
        Thread.__init__(self)
        self.killed = False
        self.lock   = Lock()
        global all_threads
        all_threads.append(self)
            
            
    def info(self, message:str) :
        """
        Safely print message to stdout by acquiring and releasing internal thread lock

        Inputs:

            >  message, str
               message to print to stdout
        """
        self.lock.acquire()
        sys.stdout.write(f"{message}")
        sys.stdout.flush()
        self.lock.release()
        

    def kill(self, killed:bool=True, verbose:bool=False) :
        """
        Set internal killed flag to the value provided.

        Inputs:

            >  killed, bool, default = True
               value to set internal killed flag to

            >  verbose, bool, default = True
               whether to safely print a message upon killing the thread
        """
        self.killed = killed
        if not self.killed and killed and verbose :
            self.info(f"Thread {self.name}.killed flag set to {killed}\n")
        
        

###===================================###
###   WorkerThread class definition   ###
###===================================###

class WorkerThread(BaseThread):
    
    def __init__(self, num_proc:int, num_results_per_proc:int, func, func_args:list=[], queue_check_freq:float=1.) :
        """
        Thread to spawn a number of parallel subprocesses and collect the results.
        Each subprocess appends results to Queue object, which is continuously checked and emptied by this thread to prevent deadlock when it is full.
        Results are transferred to an internal list (self.results) to be read by main process.

        Inputs:

            >  num_proc, int
               number of subprocesses to create

            >  num_results_per_proc, int
               number of results to be created by each subprocess

            >  func, callable 
               function to call in each process with arguments: proc_idx:int, num_results_per_proc:int, out_queue:Queue, func_args:list

            >  func_args, list, default=[] 
               custom arguments required by callable func

            >  queue_check_freq, float, default=1.
               frequency at which to check and empty Queue object
        """
        BaseThread.__init__(self)
        self.num_proc             = num_proc
        self.num_results_per_proc = num_results_per_proc
        self.queue_check_freq     = queue_check_freq
        self.func                 = func
        self.func_args            = func_args
        self.results              = []
        

    def run(self) :
        """
        Spawn subprocesses and keeping reading results from the Queue until all expected results are present.
        Loop is also exited if self.killed flag is set.
        Each subprocess runs the methof self.func with arguments: proc_idx:int, num_results_per_proc:int, out_queue:Queue, func_args:list.
        Each subprocess is expected to append self.num_results_per_proc results to the Queue.
        """

        ##  Reset state: self.killed flag to False and self.results to empty list
        self.results = []
        self.kill(False)

        ##  Create Queue to pass results from subprocesses back to here
        out_queue = Queue()

        ##  Create and start list of subprocesses configured to run self.func with arguments: proc_idx:int, num_results_per_proc:int, out_queue:Queue, func_args:list
        processes = []
        for proc_idx in range(self.num_proc) :
            proc_args = [proc_idx, self.num_results_per_proc, out_queue, self.func_args]
            p = Process(target=self.func, args=proc_args)
            p.start()
            processes.append(p)

        ##  Keep reading results from the Queue until self.results reaches the expected length, 
        exp_length = self.num_proc*self.num_results_per_proc
        while len(self.results) < exp_length and not self.killed :
            time.sleep(self.queue_check_freq)
            while not out_queue.empty() :
                self.results.append(out_queue.get())
        
        # Don't need to call process.join() because we are manually waiting for results to appear, or for
        #   processes to be interrupted
       
    

###====================================###
###   MonitorThread class definition   ###
###====================================###

class MonitorThread(BaseThread):
    
    def __init__(self, worker:WorkerThread, frequency:float=1.):
        """
        Report on the progress of the Worker thread provided in filling up worker.results.

        Inputs:

            > worker, WorkerThread
              worker thread to be monitored

            > frequency, float, default = 1.
              frequency with which to check and print the progress
        """
        BaseThread.__init__(self)
        self.worker    = worker
        self.frequency = frequency
        

    def run(self):
        """
        Keep printing updates on the progress of worker until worker.results reaches desired length, or the thread is killed.
        """

        ##  Reset kill state
        self.kill(False)

        ##  Find the target length of worker.results
        exp_length = self.worker.num_proc * self.worker.num_results_per_proc

        ##  Check and print progress updates with the configured frequency
        start_time = time.time()
        self.info(f"\rGenerating {exp_length} results")
        while len(self.worker.results) < exp_length and not self.killed :
            time.sleep(self.frequency)
            if not self.killed :
                self.info(f"\rGenerated {len(self.worker.results)} / {exp_length} results [t={time.time()-start_time:.2f}s]")

        ##  Print a final message reporting on whether generation was complete, or the monitor was killed
        if len(self.worker.results) < exp_length :
            self.info(f"Monitor killed [t={time.time()-start_time:.2f}s] [n={len(self.worker.results)}]\n")
        else :
            self.info(f"\nGeneration complete [t={time.time()-start_time:.2f}s] [n={len(self.worker.results)}]\n")
    
    

###======================###
###   Method defitions   ###
###======================###


def generate_from_processes(func, func_args:list=[], num_proc:int=1, num_results_per_proc:int=1, mon_freq:float=1.) :
    """
    Create worker and monitor threads to generate a number of datapoints using the method func, and return the results.

        Inputs:

            >  func, callable 
               function to call in each process with arguments: proc_idx:int, num_results_per_proc:int, out_queue:Queue, func_args:list

            >  func_args, list, default=[] 
               custom arguments required by callable func

            >  num_proc, int
               number of subprocesses to create

            >  num_results_per_proc, int
               number of results to be created by each subprocess

            >  mon_freq, float, default=1.
               frequency at which to print progress updates, if -ve then do not create a monitor thread
    """

    ##  Figure out whether monitor thread is requested
    use_monitor = mon_freq > 0
    
    ##  Create worker and monitor processes
    worker  = WorkerThread(num_proc, num_results_per_proc, func, func_args=func_args)
    if use_monitor : monitor = MonitorThread(worker, frequency=mon_freq)

    ##  Begin processes in parallel (monitor first)
    if use_monitor : monitor.start()
    worker .start()

    ##  Wait for processes to complete (worker first)
    worker .join()
    if use_monitor : monitor.join()
    
    ##  Return results from worker process
    return worker.results



def kill_threads(threads:list=None, verbose:bool=False) :
    """
    Call kill() method for all custom threads in the list provided. If no list then apply to all custom threads created thus far.

    Inputs:

        >  threads, list, default = None
           list of all custom threads to kill

        >  verbose, bool, default = False
           whether to print a message to stdout when thread is killed
    """

    ##  If threads is None then use global list all_threads
    if not threads :
        global all_threads
        threads = all_threads

    ##  Loop over threads and call internal kill() method
    for thread in threads :
        if not hasattr(thread, "kill") :
            continue
        thread.kill(verbose=verbose)