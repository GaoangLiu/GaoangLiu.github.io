from absl import logging

class OverwriteLog():
    # overwrite logging for kaggle kernel
    def info(self, msg):
        print(msg)
    
logging = OverwriteLog()    
# logging.set_verbosity(logging.INFO)
logging.info('Hello')

