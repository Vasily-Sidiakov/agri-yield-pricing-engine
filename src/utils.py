import yaml
import sys
import time
import threading

def load_config(config_path='config/commodities.yaml'):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

class LoadingSpinner:
    """
    A context manager that displays a '...' animation in the terminal
    while a long-running process executes in the background.
    """
    def __init__(self, message="Processing", delay=0.5):
        self.message = message
        self.delay = delay
        self.stop_running = False
        self.thread = threading.Thread(target=self._animate)

    def __enter__(self):
        self.stop_running = False
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        self.thread.join()
        # Clean the line after finishing
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def _animate(self):
        dots = [".  ", ".. ", "..."]
        i = 0
        while not self.stop_running:
            sys.stdout.write('\r' + f"   {self.message} {dots[i % 3]}")
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1