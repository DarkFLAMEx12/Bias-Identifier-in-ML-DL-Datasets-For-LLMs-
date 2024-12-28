import tkinter as tk
from tkinter import *
import sv_ttk
from tkinter import ttk
from Functions_ import *

LBD = tk.Tk()
sv_ttk.set_theme("dark")

#functions
def run():
    filepath = Attach_Dataset.get()
    with open(filepath, "r") as file:
        document = file.read()
    
    modified_sentences = analyze_and_modify_bias_in_text(document, model, vectorizer=cv)

    modified_df = pd.DataFrame({
        'modified_sentence': modified_sentences
    })

    modified_df.to_csv('modified_dataset.csv', index=False)
        

title = ttk.Label(text="Bias Identifier in NLP Datasets (For LLMs)")
title.grid(row=0, column=0, padx=20, pady=20)
title.config(font=("Arial", 15, "bold"))

label1 = ttk.Label(text="By DarkFLAME Corporation")
label1.grid(row=0, column=1, padx=20, pady=20, sticky="ne")
label1.config(font=("Arial", 5, "bold"))

label2 = ttk.Label(text="Please Attach The Dataset, The File Path:")
label2.grid(row=1, column=0, padx=20, pady=20, sticky = "nw")

Attach_Dataset = ttk.Entry(text="File Path:")
Attach_Dataset.grid(row=1, column=1, padx=20, pady=20, sticky = "ne")


Run = ttk.Button(text = "Save The Modified Dataset:", command = run, style = "Accent.TButton")
Run.grid(row = 1, column=2, padx = 20, pady=20)



LBD.mainloop()
