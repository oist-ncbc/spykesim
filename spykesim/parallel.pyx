from tqdm import tqdm, tqdm_notebook
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor, as_completed

def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger = getLogger()
    if (logger.hasHandlers()):
        logger.handlers.clear()    
    logger.addHandler(handler)
    logger.setLevel(INFO)

def parallel_process(array, function, n_jobs=48, use_kwargs=False, front_num=2, notebook=False):
    """
        A parallel version of the map function with a progress bar. 
        The original function is retrieved from http://danshiebler.com/2016-09-14-parallel-progress-bar/
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    # Map filter reduceを使って書き直したい
    tqdm_ = tqdm_notebook if notebook else tqdm
    init_logger()
    getLogger().info(
        "Execution of a function {} starts".format(
            getattr(
                function, "__name__", "Unknown")
        )
    )
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm_(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        getLogger().info("submit end")
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        getLogger().info("Progress of the calculation")
        for f in tqdm_(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    getLogger().info("Progress of the aggregation")
    for i, future in tqdm_(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    getLogger().info("calculation end")
    return front + out
