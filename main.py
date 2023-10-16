import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import tensorflow as tf
from DeepCNN import DeepCNN
import numpy as np


class App(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self)
        with open('NLP-Project-Sentiment/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Define NN settings:
        self.VOCAB_SIZE = len(self.tokenizer.word_index)
        self.MAX_LENGTH = 52
        self.EMB_DIM = 200
        self.NB_FILTERS = 100
        self.FFN_UNITS = 256
        self.NB_CLASSES = 2
        self.DROUPOUT_RATE = 0.2
        self.BATCH_SIZE = 32
        self.NB_EPOCHS = 5
        self.window = parent
        self.deepCNN = DeepCNN(vocab_size=self.VOCAB_SIZE,
                               emb_dim=self.EMB_DIM,
                               nb_filters=self.NB_FILTERS,
                               FFN_units=self.FFN_UNITS,
                               nb_classes=self.NB_CLASSES,
                               dropout_rate=self.DROUPOUT_RATE,
                               )
        checkpoint_path = "NLP-Project-Sentiment"
        ckpt = tf.train.Checkpoint(deepCNN=self.deepCNN)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

        if (ckpt_manager.latest_checkpoint):
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest Checkpoint restored")
        self.setup_widgets()
        self._offsetx = 0
        self._offsety = 0
        self.bind('<Button-1>', self.clickwin)
        self.bind('<B1-Motion>', self.dragwin)

    def dragwin(self, event):
        x = super().winfo_pointerx() - self._offsetx
        y = super().winfo_pointery() - self._offsety
        self.window.geometry(f"+{x}+{y}")

    def clickwin(self, event):
        self._offsetx = super().winfo_pointerx() - super().winfo_rootx()
        self._offsety = super().winfo_pointery() - super().winfo_rooty()

    def processInput(self, text):
        # Convert the input text to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=self.MAX_LENGTH,
                                                                        padding='post',
                                                                        truncating='post')
        # Predict using the sequence
        prediction = self.deepCNN(np.array(padded_sequence), training=False)
        # Convert model's output to a user-friendly result
        if prediction >= 0.5:
            return f"The result is Positive, {prediction} confidence"
        else:
            return f"The result is Negative, {1 - prediction} confidence"

    def showResult(self):
        text = self.search_entry.get()
        result = self.processInput(text)
        messagebox.showinfo("Result", result)

    def setup_widgets(self):
        # Close Button
        self.close_button = ttk.Button(self, text="X", command=self.quit)
        self.close_button.grid(row=0, column=1, sticky="ne")  # Positioned at the top right corner

        # Text input label
        self.search_label = ttk.Label(
            self,
            text="Enter Text:",
            justify="center",
            font=("-size", 10, "-weight", "bold"),
        )
        self.search_label.grid(row=1, column=0, padx=5, pady=(0, 10), sticky="ew")

        # Text input field
        self.search_entry = ttk.Entry(self)
        self.search_entry.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="ew")

        # Button to submit the text
        self.submit_button = ttk.Button(self, text="Submit", command=self.showResult)
        self.submit_button.grid(row=2, columnspan=2, pady=(0, 10))

        # Making the input area and button dynamically change size based on app size
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)


window = tk.Tk()
window.title("Sentimental Analysis")

# Calculate the screen's width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the dimensions of the main window
window_width = 800
window_height = 500

# Calculate the x and y coordinates to center the window
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

window.geometry(f"{window_width}x{window_height}+{x}+{y}")  # Center the window on the screen

window.tk.call("source", "azure.tcl")
window.tk.call("set_theme", "dark")
window.overrideredirect(True)

app = App(window)
app.pack(fill="both", expand=True)

window.mainloop()
