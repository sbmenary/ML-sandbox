###
###  connect4.parallel.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of parallelisation methods
"""

import sys, time
from threading import Lock, Thread

from multiprocess import set_start_method, Process, Queue



all_threads = []
        

        
class BaseThread(Thread) :
    
    def __init__(self):
        Thread.__init__(self)
        self.killed = False
        self.lock   = Lock()
        global all_threads
        all_threads.append(self)
        
    def kill(self, killed=True, verbose=False) :
        if not self.killed and killed and verbose :
            self.info(f"Killing thread: {self.name}\n")
        self.killed = killed
            
    def info(self, message) :
        self.lock.acquire()
        sys.stdout.write(f"{message}")
        sys.stdout.flush()
        self.lock.release()
        
        
class WorkerThread(BaseThread):
    
    def __init__(self, num_proc, num_results_per_proc, func, func_args=[], queue_check_freq=1.):
        BaseThread.__init__(self)
        self.num_proc             = num_proc
        self.num_results_per_proc = num_results_per_proc
        self.queue_check_freq     = queue_check_freq
        self.func                 = func
        self.func_args            = func_args
        self.results              = []
        
    def run(self):
        self.kill(False)
        out_queue = Queue()

        processes = []
        for proc_idx in range(self.num_proc) :
            proc_args = [proc_idx, self.num_results_per_proc, out_queue, self.func_args]
            p = Process(target=self.func, args=proc_args)
            p.start()
            processes.append(p)

        exp_length = self.num_proc*self.num_results_per_proc
        while len(self.results) < exp_length and not self.killed :
            time.sleep(self.queue_check_freq)
            while not out_queue.empty() :
                self.results.append(out_queue.get())
        
        # Don't need to call process.join() because we are manually waiting for results to appear, or for
        #   processes to be interrupted
       
    
class MonitorThread(BaseThread):
    
    def __init__(self, worker, frequency=1):
        BaseThread.__init__(self)
        self.worker    = worker
        self.frequency = frequency
        
    def run(self):
        self.kill(False)
        exp_length = self.worker.num_proc*self.worker.num_results_per_proc
        start_time = time.time()
        self.info(f"\rGenerating {exp_length} results")
        while len(self.worker.results) < exp_length and not self.killed :
            time.sleep(self.frequency)
            if not self.killed :
                self.info(f"\rGenerated {len(self.worker.results)} / {exp_length} results [t={time.time()-start_time:.2f}s]")
        if len(self.worker.results) < exp_length :
            self.info(f"\nMonitor killed [t={time.time()-start_time:.2f}s] [n={len(self.worker.results)}]\n")
        else :
            self.info(f"\nGeneration complete [t={time.time()-start_time:.2f}s] [n={len(self.worker.results)}]\n")
    
    

def generate_from_processes(func, func_args=[], num_proc:int=1, num_results_per_proc:int=1, mon_freq:int=1.) :
    """
    Method generate_from_processes
    """
    
    ##  Create worker and monitor processes
    worker  = WorkerThread(num_proc, num_results_per_proc, func, func_args=func_args)
    monitor = MonitorThread(worker, frequency=mon_freq)

    ##  Begin processes in parallel (monitor first)
    monitor.start()
    worker .start()

    ##  Wait for processes to complete (worker first)
    worker .join()
    monitor.join()
    
    ##  Return results from worker process
    return worker.results



def kill_threads(threads=None, verbose=False) :
    """
    Method kill_threads
    """
    if type(threads) is type(None) :
        global all_threads
        threads = all_threads
    for thread in threads :
        if not hasattr(thread, "kill") :
            continue
        thread.kill(verbose=verbose)