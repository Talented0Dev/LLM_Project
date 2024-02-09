import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import ebooklib
from ebooklib import epub
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# https://www.upwork.com/ab/proposals/job/~01cbcbb64654457e34/apply/
# pip install torch torchvision torchaudio
class NovelGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Novel Generator")
        self.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Theme and Synopsis
        self.theme_label = ttk.Label(self, text="Theme:")
        self.theme_label.grid(row=0, column=0, sticky="w")
        self.theme_entry = ttk.Entry(self, width=50)
        self.theme_entry.grid(row=0, column=1, padx=5, pady=5)

        self.synopsis_label = ttk.Label(self, text="Synopsis:")
        self.synopsis_label.grid(row=1, column=0, sticky="w")
        self.synopsis_text = scrolledtext.ScrolledText(self, width=50, height=5)
        self.synopsis_text.grid(row=1, column=1, padx=5, pady=5)

        # Number of Words and Chapters
        self.words_label = ttk.Label(self, text="Number of Words:")
        self.words_label.grid(row=2, column=0, sticky="w")
        self.words_entry = ttk.Entry(self)
        self.words_entry.grid(row=2, column=1, padx=5, pady=5)

        self.chapters_label = ttk.Label(self, text="Number of Chapters:")
        self.chapters_label.grid(row=3, column=0, sticky="w")
        self.chapters_entry = ttk.Entry(self)
        self.chapters_entry.grid(row=3, column=1, padx=5, pady=5)

        # Generate Button
        self.generate_button = ttk.Button(self, text="Generate Novel", command=self.generate_novel)
        self.generate_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Generated Novel Display
        self.generated_novel_label = ttk.Label(self, text="Generated Novel:")
        self.generated_novel_label.grid(row=5, column=0, sticky="w")
        self.generated_novel_text = scrolledtext.ScrolledText(self, width=80, height=20)
        self.generated_novel_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

    def generate_novel(self):
        # Get user input
        theme = self.theme_entry.get()
        synopsis = self.synopsis_text.get("1.0", tk.END)
        num_words = int(self.words_entry.get())
        num_chapters = int(self.chapters_entry.get())

        # Generate novel content using ChatGPT
        novel_content = self.generate_novel_content(theme, synopsis, num_words, num_chapters)

        # Display generated novel
        self.generated_novel_text.delete("1.0", tk.END)
        self.generated_novel_text.insert(tk.END, novel_content)
    

    def generate_novel_content(self, theme, synopsis, num_words, num_chapters):
        # Use GPT-2 model for text generation
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Construct a prompt string based on user input
        prompt = f"{theme}. {synopsis}. One day, he experienced a life-changing event. Write a {num_words}-word chapter."

        # Generate novel content using the model
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        max_length = num_words + 100  # Add extra tokens for safety
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_chapters,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,  # Example: Use beam search with 5 beams
            early_stopping=True
        )

        # Decode generated sequences
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text
if __name__ == "__main__":
    app = NovelGeneratorApp()
    app.mainloop()