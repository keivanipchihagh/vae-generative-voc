##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

from colorama import init as colorama_init
from colorama import Fore, Style


class Logger(object):

    def __init__(self) -> None:
        colorama_init()


    def info(self, msg: str) -> None:
        print(f"{Fore.CYAN}INFO) {msg}{Style.RESET_ALL}")


    def warn(self, msg: str) -> None:
        print(f"{Fore.YELLOW}INFO) {msg}{Style.RESET_ALL}")


    def error(self, msg: str) -> None:
        print(f"{Fore.RED}INFO) {msg}{Style.RESET_ALL}")
