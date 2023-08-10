import logging
import os
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler


class Log(object):
    def __init__(self):
        self.logFilePath = os.getenv('LOG_PATH', 'logs')
        self.infoLogFilePath = '{}/info.log'.format(self.logFilePath)
        self.errorLogFilePath = '{}/error.log'.format(self.logFilePath)
        os.makedirs(self.logFilePath, exist_ok=True)
        self.logger = self.initLogger()
        # 如果不存在就创建一个新的日志文件夹


    def initLogger(self):
        logger = logging.getLogger('flask')

        # formater,info和error公用
        dateFmt = '%Y-%m-%d %H:%M:%S'
        logFormater = '%(asctime)s %(levelname)s process-%(process)s thread-%(thread)s %(name)s-%(filename)s:%(lineno)d %(message)s'
        simpleFormater = logging.Formatter(logFormater, datefmt=dateFmt)

        # info 日志文件输出
        infoFileHandler = TimedRotatingFileHandler(self.infoLogFilePath, 'D')
        infoFileHandler.setLevel(logging.INFO)
        infoFileHandler.setFormatter(simpleFormater)

        # error 日志文件输出
        errorFileHandler = TimedRotatingFileHandler(self.errorLogFilePath, 'D')
        errorFileHandler.setLevel(logging.ERROR)
        errorFileHandler.setFormatter(simpleFormater)

        # 输出到终端
        consoleHandler = StreamHandler(sys.stdout)
        consoleHandler.setLevel("DEBUG")
        consoleHandler.setFormatter(simpleFormater)

        logger.addHandler(infoFileHandler)
        logger.addHandler(errorFileHandler)
        logger.addHandler(consoleHandler)

        return logger

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def fatal(self, message):
        self.logger.fatal(message)


logging = Log()
