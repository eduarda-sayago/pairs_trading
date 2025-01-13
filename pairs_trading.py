import tensorflow as tf
import subprocess

# Check for GPU availability
if tf.test.gpu_device_name():
    print(f"GPU found: {tf.test.gpu_device_name()}")
else:
    print("No GPU found.")

# Execute rnn_trainer.py
try:
    subprocess.run(["python", "rnn_trainer.py"], check=True)
    print("rnn_trainer.py executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error while executing rnn_trainer.py: {e}")
