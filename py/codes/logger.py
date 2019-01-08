import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d %H:%M:%S %p"

logging.basicConfig(filename='hupu.log', level=logging.INFO, format = LOG_FORMAT, datefmt= DATE_FORMAT)

logging.debug('debug message')
logging.info('info message')
logging.warning('warn message')
logging.error('error message')
logging.critical('critical message')


