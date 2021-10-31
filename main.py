from tkinter import filedialog, Canvas
from functions import readImagesTrain, readImage, processImage, readTrainedFile, saveTrainedFile, calc_deltas, join_path, PATH
import tkinter as tk
import os

### FUNCTIONS ###


def config(bg, text, state):
    warnings.configure(bg=bg)
    warnings.itemconfig(warningsTextId, text=text)
    runApps.configure(state=state)


def train():
    dict = readTrainedFile()
    if readTrainedFile() is not None:
        return dict
    dict = readImagesTrain()
    saveTrainedFile(dict)
    config("green", "Training successfully!!!", tk.ACTIVE)


def predict():
    filename = filedialog.askopenfilename(
        initialdir="/",
        title="Select File",
        filetypes=(("all files", "*.*"), ("jpg", "*.jpg"), ("png", "*.png"))
    )
    if (filename is None or filename == ""):
        answer = "Can't find answer"
    else:
        dict_vec = readTrainedFile()
        image = readImage(filename)
        image = processImage(image)
        dict_delta = calc_deltas(dict_vec, image)
        answer = min(dict_delta, key=dict_delta.get)
    answerCanVas.itemconfig(answerCanVasId, text=answer)


#################

root = tk.Tk()
root.title("do an suy dien mo")
root.geometry("500x280")

warnings = Canvas(root, width=200, height=100)
warningsTextId = warnings.create_text(
    100, 50, fill="black", font=('Helvetica 15 bold'))
warnings.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8)
frame.pack()

training = tk.Button(frame, bg="#263D42", fg="white",
                     text="Train model", padx=10, pady=5, command=train)
training.pack(padx=5, pady=5, side=tk.LEFT)

runApps = tk.Button(frame, bg="#263D42", fg="white",
                    text="Predict Images", padx=10, pady=5, command=predict)
runApps.pack(padx=5, pady=5, side=tk.RIGHT)

if os.path.exists(join_path(PATH, 'model/model.h5')):
    config("green", "Training successfully!!!", tk.ACTIVE)
else:
    config("yellow", "Need training!!!", tk.DISABLED)

answerCanVas = Canvas(root, width=500, height=100, bg="SpringGreen2")
answerCanVasId = answerCanVas.create_text(
    250, 50, fill="black", font=('Helvetica 15 bold'))
answerCanVas.pack()

root.mainloop()
