"""
Creates a GUI to interact with the LLM model.
"""

import tkinter as tk
from tkinter import scrolledtext
from app.main import vector_index, llm

class LLMapp:
    """
    Class for the GUI instance
    """
    def __init__(self, root):
        """
        Initializes the GUI for the LLM
        """
        self.root = root
        self.root.title = "LLM Query Application"

        self.query_label = tk.Label(root, text= "Enter Question:")
        self.query_label.pack()

        self.query = tk.Entry(root, width= 50)
        self.query.pack()

        self.submit_button = tk.Button(root, text= "Submit", command= self.ask_the_app)
        self.submit_button.pack()

        self.response = scrolledtext.ScrolledText(root, wrap= tk.WORD, width= 50, height= 10)
        self.response.pack()

        self.exit_button = tk.Button(root, text= "Quit Application", command= self.exit_the_app)
        self.exit_button.pack()
    
    def ask_the_app(self):
        """
        Returns the result of the question from the LLM in main.py
        """
        query = self.query.get()
        if query:
            response = vector_index.query(query, llm).strip()
            self.response.delete('1.0', tk.END)
            self.response.insert(tk.END, response)
        else:
            # Insert a message asking the user to enter a valid query
            self.response.delete('1.0', tk.END)
            self.response.insert(tk.END, "Please enter a valid query")
    
    def exit_the_app(self):
        """
        Exits the application
        """
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMapp(root)
    root.mainloop()
