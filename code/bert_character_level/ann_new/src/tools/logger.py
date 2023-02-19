import logging

from tools.utils import only_main_process


class Logger:
    def __init__(self, app_name, *, format_str=None, file_path=None):
        self.logger = None
        self.init(app_name, format_str, file_path)

    @only_main_process
    def init(self, app_name, format_str, file_path):
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)

        if format_str is not None:
            formatter = logging.Formatter(format_str)
        else:
            formatter = logging.Formatter()

        if file_path is not None:
            fh = logging.FileHandler(file_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @only_main_process
    def info(self, s):
        self.logger.info(s)
