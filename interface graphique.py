import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import filedialog
import requests

# Importe la fonction de calcul de similarite
from similarity_calculation import calculate_similarity, load_text_database

class TextSimilarityApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Text Similarity Calculator")

        self.text_area = scrolledtext.ScrolledText(master, width=50, height=10, wrap=tk.WORD)
        self.text_area.pack(pady=10)

        self.load_database_button = tk.Button(master, text="Load Database", command=self.load_database)
        self.load_database_button.pack(pady=5)

        self.add_url_button = tk.Button(master, text="Add URL", command=self.add_url)
        self.add_url_button.pack(pady=5)

        self.upload_file_button = tk.Button(master, text="Upload File", command=self.upload_file)
        self.upload_file_button.pack(pady=5)

        self.calculate_similarity_button = tk.Button(master, text="Calculate Similarity", command=self.calculate_similarity)
        self.calculate_similarity_button.pack(pady=5)

        self.result_label = tk.Label(master, text="")
        self.result_label.pack(pady=5)

    def load_database(self):
        directory_path = filedialog.askdirectory()
        self.text_database = load_text_database(directory_path)
        messagebox.showinfo("Database Loaded", "Database loaded successfully!")

    def add_url(self):
        url = simpledialog.askstring("Add URL", "Enter the URL:")
        if url:
            try:
                response = requests.get(url)
                self.text_area.insert(tk.END, response.text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch text from URL: {e}")

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                text = file.read()
                self.text_area.insert(tk.END, text)

    def calculate_similarity(self):
        input_text = self.text_area.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showerror("Error", "Please enter text to calculate similarity.")
            return

        if not hasattr(self, 'text_database'):
            messagebox.showerror("Error", "Please load the database first.")
            return

        similarities = calculate_similarity(input_text, self.text_database)
        # Affiche les resultats
        self.result_label.config(text="Similarity calculated.")

root = tk.Tk()
app = TextSimilarityApp(root)
root.mainloop()




