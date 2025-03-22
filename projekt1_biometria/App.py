import tkinter as tk
import customtkinter
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from numpy.ma.core import negative


class SidePanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky="ns")

        self.brightness = ctk.CTkLabel(self, text='brightness')
        self.brightness.grid(row=0, column=0, sticky="nsew")
        self.brightness_slider = ctk.CTkSlider(self, from_=-255, to=255, number_of_steps=100)
        self.brightness_slider.grid(row=1, column=0, pady=10)

        self.contrast = ctk.CTkLabel(self, text='Contrast')
        self.contrast.grid(row=2, column=0, sticky="nsew")
        self.contrast_slider = ctk.CTkSlider(self, from_=-255, to=255, number_of_steps=100)
        self.contrast_slider.grid(row=3, column=0, pady=10)

        self.label3 = ctk.CTkLabel(self, text='costam')
        self.label3.grid(row=4, column=0, sticky="nsew")
        self.slider3 = ctk.CTkSlider(self, from_=-255, to=255, number_of_steps=100)
        self.slider3.grid(row=5, column=0, pady=10)

class RightPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=2, rowspan=2, padx=20, pady=20, sticky="nsew")

        self.negative_checkbox = customtkinter.CTkCheckBox(self, text='negative')
        self.negative_checkbox.grid(row=0, column=0, sticky="nsew", pady=10, padx=10)

        self.black_and_white = customtkinter.CTkCheckBox(self, text='black and white')
        self.black_and_white.grid(row=1, column=0, sticky="nsew", pady=10, padx=10)

class MainPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.side_panel = SidePanel(self)
        self.side_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.right_panel = RightPanel(self)
        self.right_panel.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

        # ma być możliwość cofnięcia zmiany, więc trzeba będzie zrobić jakąś liste gdzie jest przechowywane np
        # 5 ostatnich wersji obrazu
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=1, pady=10, padx=10, sticky="ew")
        self.undo_btn = ctk.CTkButton(self.top_frame, text="Undo", width=60)
        self.undo_btn.pack(side="left", padx=5)
        self.redo_btn = ctk.CTkButton(self.top_frame, text="Redo", width=60, state='disabled')
        self.redo_btn.pack(side="left", padx=5)

        # w tej ramce wyświetla się zdjęcie
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.select_btn = ctk.CTkButton(self.image_frame, text="Wybierz obraz", command=self.load_image)
        self.select_btn.grid(row=0, column=1, pady=20, padx=20, sticky="nsew")

        #na tym rysuje się zdjęcie
        self.canvas = ctk.CTkCanvas(self.image_frame, bg="black", width=500, height=300)
        self.canvas.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # tutaj hsitogram
        self.histogram_frame = ctk.CTkFrame(self.image_frame)
        self.histogram_frame.grid(row=2, column=1, pady=10)

        self.button_save = ctk.CTkButton(self.image_frame, text="Zapisz rezultat", command=self.save_image)
        self.button_save.grid(row=3, column=1, pady=20)

        self.original_image_array = None
        self.processed_image_array = None

        self.side_panel.brightness_slider.configure(command=self.update_image)
        self.right_panel.negative_checkbox.configure(command=self.update_image)
        self.right_panel.black_and_white.configure(command=self.update_image)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.original_image_array= np.array(Image.open(file_path))
            self.processed_image_array = self.original_image_array.copy()
            self.display_image()
            self.display_histogram()

    def update_image(self, event=None):
        image = self.original_image_array.copy()

        brightness = self.side_panel.brightness_slider.get()
        image = self.adjust_brigthness(image, brightness)

        if self.right_panel.negative_checkbox.get():
            image = self.negative(image)

        if self.right_panel.black_and_white.get():
            image = self.to_greyscale(image)

        self.processed_image_array = image
        self.display_image()

    def adjust_brigthness(self, image, brightness):
        return np.clip(image + brightness, 0, 255).astype(np.uint8)

    def negative(self, image):
        return 255 - image

    def to_greyscale(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


    def display_image(self):
        max_width, max_height = 500, 300
        img_height, img_width = self.processed_image_array.shape[:2]
        scale = min(max_width / img_width, max_height / img_height)
        img_width = int(img_width * scale)
        img_height = int(img_height * scale)

        img_pil = Image.fromarray(np.uint8(self.processed_image_array))
        resized_image = img_pil.resize((img_width, img_height))
        img_tk = ImageTk.PhotoImage(resized_image)

        self.canvas.delete("all")
        self.canvas.config(width=max_width, height=max_height)
        x_offset = max_width  / 2
        y_offset = max_height / 2

        self.canvas.create_image(x_offset, y_offset, anchor=tk.CENTER, image=img_tk)
        self.canvas.image = img_tk

    def save_image(self):
        if self.processed_image_array is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                         filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                saved_image = Image.fromarray(self.processed_image_array)
                saved_image.save(file_path)

    def display_histogram(self):
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()

        if len(self.processed_image_array.shape) == 3:

            r_histogram, r_bins = np.histogram(self.processed_image_array[..., 0], bins=256, range=(0,255))
            g_histogram, g_bins = np.histogram(self.processed_image_array[..., 1], bins=256, range=(0, 255))
            b_histogram, b_bins = np.histogram(self.processed_image_array[..., 2], bins=256, range=(0, 255))

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(r_bins[:-1], r_histogram, width=1, color='red', alpha=0.5, label='Red')
            ax.bar(g_bins[:-1], g_histogram, width=1, color='green', alpha=0.5, label='Green')
            ax.bar(b_bins[:-1], b_histogram, width=1, color='blue', alpha=0.5, label='Blue')
            ax.set_xlim(0,255)
            ax.legend()
        else:
            gray_histogram, bins = np.histogram(self.processed_image_array, bins=256, range=(0,255))
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(bins[:-1], gray_histogram, width=1,color='black',label='Grey')
            ax.set_xlim(0,255)

        canvas = FigureCanvasTkAgg(fig, master=self.histogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Processor")
        self.geometry("1000x800")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.main_panel = MainPanel(self)



if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
