import sys
import subprocess
import re
from threading import Thread

def main():    
    script_descriptor = open("run.py")
    a_script = script_descriptor. read()
    sys. argv = ["run.py", "27"]
    exec(a_script)
    script_descriptor. close()
    
    sys. argv = ["run.py", "28"]
    exec(a_script)
    script_descriptor. close()
    
if __name__ == "__main__":
    main()