
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import tkinter as tk

# Initialize direct_responses globally
direct_responses = {}


# GUI Related Functions
def send(event=None):  # Event is passed by binders.
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("1.0", "end")

    if msg != "":
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = get_response(msg)
        ChatLog.insert(tk.END, "Moe: " + res + "\n\n")

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)


def normalize_sentence(sentence):
    """Function to preprocess the sentence."""
    sentence = sentence.lower()  # Lowercase
    return " ".join(tokenize(sentence))  # Tokenize and join back into a string


def is_direct_match(user_input, pattern):
    """Function to determine if there is a match."""
    # Example simple matching, can be replaced with more complex logic
    return user_input in pattern or pattern in user_input


# Load model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(r"E:\\chatbot-Ai240\\Data.json", "r") as json_data:
    Data = json.load(json_data)

# Populating the direct_responses dictionary
    # with direct user input-response mappings
for intent in Data["intents"]:
    if intent["tag"] == "ai_course_query":
        for pattern in intent["patterns"]:
            direct_responses[pattern] = random.choice(intent["responses"])

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(sentence):
    if sentence.lower() == "quit":
        base.destroy()

    normalized_sentence = normalize_sentence(sentence)

    # Check for a direct response first
    matched_response = None
    for pattern in direct_responses:
        if is_direct_match(normalized_sentence, pattern):
            matched_response = direct_responses[pattern]
            break

    if matched_response:
        return matched_response

    # Proceed with the intent recognition model
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in Data["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return "I do not understand..."


# Creating GUI with tkinter
base = tk.Tk()
base.title("Chatbot by Abdalrahman Alshamaileh")
base.geometry("800x800")  # Larger window size
base.attributes("-fullscreen", True)  # Make the window full-screen

# Create Chat window
ChatLog = tk.Text(base, bd=0, bg="white", font=("Arial", 12))
ChatLog.config(state=tk.DISABLED, wrap=tk.WORD)
scrollbar = tk.Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog["yscrollcommand"] = scrollbar.set

# Bind scrollbar to Chat window
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
ChatLog.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create Button to send message
SendButton = tk.Button(
    base,
    font=("Verdana", 12, "bold"),
    text="Send",
    command=send,
    bd=0,
    bg="#32de97",
    activebackground="#3c9d9b",
    fg="#ffffff",
)

# Create the box to enter message
EntryBox = tk.Text(base, bd=0, bg="white", font=("Arial", 12), height=3)
EntryBox.bind("<Return>", send)

# Place components on the screen
SendButton.pack(side=tk.BOTTOM, fill=tk.X)
EntryBox.pack(side=tk.BOTTOM, fill=tk.X)

# Credit label
credit_label = tk.Label(
    base, text="Worked by: Abdalrahman Alshamaileh", font=("Arial", 10), fg="grey"
)
credit_label.pack(side=tk.BOTTOM, fill=tk.X)

base.mainloop()
