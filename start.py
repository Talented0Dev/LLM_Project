import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from transformers import BertForQuestionAnswering, BertTokenizer

import torch

class LLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM Question Answering App")
        self.geometry("600x400")

        self.create_widgets()

    def create_widgets(self):
        self.question_label = tk.Label(self, text="Enter your question:")
        self.question_label.pack()

        self.question_entry = tk.Entry(self, width=50)
        self.question_entry.pack()

        self.load_file_button = tk.Button(self, text="Load File", command=self.load_file)
        self.load_file_button.pack()

        self.text_display = scrolledtext.ScrolledText(self, width=60, height=15)
        self.text_display.pack()

        self.answer_label = tk.Label(self, text="Answer:")
        self.answer_label.pack()

        self.answer_display = tk.Label(self, text="")
        self.answer_display.pack()

        self.answer_button = tk.Button(self, text="Get Answer", command=self.get_answer)
        self.answer_button.pack()

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                self.text_display.delete(1.0, tk.END)
                self.text_display.insert(tk.END, text)

    def get_answer(self):
        question = self.question_entry.get()
        if not question:
            self.answer_display.config(text="Please enter a question.")
            return

        file_text = self.text_display.get(1.0, tk.END)
        if not file_text.strip():
            self.answer_display.config(text="Please load a file.")
            return

        answer = self.answer_question_from_text(file_text, question)
        self.answer_display.config(text=answer)

    def answer_question_from_text(self, text, question):
        # Specify the directory where you extracted the model files
        model_directory = "/path/to/your/model/files"

        # Load the model and tokenizer from the manually downloaded files
        model = BertForQuestionAnswering.from_pretrained(model_directory)
        tokenizer = BertTokenizer.from_pretrained(model_directory)

        inputs = tokenizer(question, text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            start_scores, end_scores = model(**inputs)

        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer_tokens = all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer

if __name__ == "__main__":
    app = LLMApp()
    app.mainloop()