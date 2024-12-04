import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer



class NovelGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Novel Generator")
        self.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        self.theme_label = ttk.Label(self, text="Theme:")
        self.theme_label.grid(row=0, column=0, sticky="w")
        self.theme_entry = ttk.Entry(self, width=50)
        self.theme_entry.grid(row=0, column=1, padx=5, pady=5)

        self.synopsis_label = ttk.Label(self, text="Synopsis:")
        self.synopsis_label.grid(row=1, column=0, sticky="w")
        self.synopsis_text = scrolledtext.ScrolledText(self, width=50, height=5)
        self.synopsis_text.grid(row=1, column=1, padx=5, pady=5)

        self.words_label = ttk.Label(self, text="Number of Words:")
        self.words_label.grid(row=2, column=0, sticky="w")
        self.words_entry = ttk.Entry(self)
        self.words_entry.grid(row=2, column=1, padx=5, pady=5)

        self.chapters_label = ttk.Label(self, text="Number of Chapters:")
        self.chapters_label.grid(row=3, column=0, sticky="w")
        self.chapters_entry = ttk.Entry(self)
        self.chapters_entry.grid(row=3, column=1, padx=5, pady=5)

        self.generate_button = ttk.Button(self, text="Generate Novel", command=self.generate_novel)
        self.generate_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.generated_novel_label = ttk.Label(self, text="Generated Novel:")
        self.generated_novel_label.grid(row=5, column=0, sticky="w")
        self.generated_novel_text = scrolledtext.ScrolledText(self, width=80, height=20)
        self.generated_novel_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

    def generate_novel(self):
        theme = self.theme_entry.get()
        synopsis = self.synopsis_text.get("1.0", tk.END)
        num_words = int(self.words_entry.get())
        num_chapters = int(self.chapters_entry.get())

        novel_content = self.generate_novel_content(theme, synopsis, num_words, num_chapters)

        self.generated_novel_text.delete("1.0", tk.END)
        self.generated_novel_text.insert(tk.END, novel_content)

    def generate_novel_content(self, theme, synopsis, num_words, num_chapters):
        novel_content = f"Theme: {theme}\nSynopsis: {synopsis}\n\n"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        for chapter in range(1, num_chapters + 1):
            novel_content += f"Chapter {chapter}:\n"
            chapter_text = self.generate_chapter_content(theme, synopsis, num_words, tokenizer, model)
            novel_content += chapter_text + "\n\n"
        return novel_content

    def generate_chapter_content(self, theme, synopsis, num_words, tokenizer, model):
        input_text = f"{theme}. {synopsis}. "
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        max_length = num_words + len(input_ids[0])
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        chapter_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return chapter_text

if __name__ == "__main__":
    app = NovelGeneratorApp()
    app.mainloop()
