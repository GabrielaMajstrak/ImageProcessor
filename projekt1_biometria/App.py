import tkinter as tk
import customtkinter
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from click import command
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from numpy.ma.core import negative


class SidePanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=0, rowspan=2, padx=20, pady=20, sticky="ns")

        self.brightness = ctk.CTkLabel(self, text='brightness')
        self.brightness.grid(row=0, column=0, sticky="nsew")
        self.brightness_slider = ctk.CTkSlider(self, from_=-255, to=255, number_of_steps=510)
        self.brightness_slider.grid(row=1, column=0, pady=10)

        self.contrast = ctk.CTkCheckBox(self, text="Contrast")
        self.contrast.grid(row=2, column=0, sticky="nsew")
        self.contrast_slider = ctk.CTkSlider(self, from_=0, to=20, number_of_steps=160)
        self.contrast_slider.set(0)
        self.contrast_slider.grid(row=3, column=0, pady=10)

        # self.threshold = ctk.CTkLabel(self, text='Binary threshold')
        # self.threshold.grid(row=4, column=0, sticky="nsew")
        # self.threshold = ctk.CTkSlider(self, from_=0, to=255, number_of_steps=255)
        # self.threshold.set(10)
        # self.threshold.grid(row=5, column=0, pady=10)

        self.vignette = ctk.CTkLabel(self, text="Vignette")
        self.vignette.grid(row=4, column=0, sticky="nsew")
        self.vignette_slider = ctk.CTkSlider(self, from_=0, to=1, number_of_steps=20)
        self.vignette_slider.set(0)
        self.vignette_slider.grid(row=5, column=0, pady=10)

class Edges_Panel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.label = ctk.CTkLabel(self, text='Edges_detection', font=ctk.CTkFont(weight="bold"))
        self.label.grid(row=0, column=0, pady=(0, 10), padx=10)

        self.roberts_cross_checkbox = ctk.CTkCheckBox(self, text="Roberts Cross")
        self.roberts_cross_checkbox.grid(row=1, column=0, pady=(0, 10))

        self.sobel_checkbox = ctk.CTkCheckBox(self, text="Sobel")
        self.sobel_checkbox.grid(row=2, column=0, pady=(0, 10))

class FilterPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text='Filters', font=ctk.CTkFont(weight="bold"))
        self.label.grid(row=0, column=0, pady=(0, 10), padx=10)

        # Averaging
        self.avg_filter_checkbox = ctk.CTkCheckBox(self, text="Averaging")
        self.avg_filter_checkbox.grid(row=1, column=0, sticky="w", pady=5, padx=10)

        self.avg_kernel_label = ctk.CTkLabel(self, text="Kernel size (odd)")
        self.avg_kernel_label.grid(row=2, column=0, sticky="w", padx=10)
        self.avg_kernel_slider = ctk.CTkSlider(self, from_=3, to=11, number_of_steps=4)
        self.avg_kernel_slider.set(3)
        self.avg_kernel_slider.grid(row=3, column=0, padx=10, pady=(0, 10))

        # Gaussian
        self.gaussian_filter_checkbox = ctk.CTkCheckBox(self, text="Gaussian")
        self.gaussian_filter_checkbox.grid(row=4, column=0, sticky="w", pady=5, padx=10)

        self.gaussian_sigma_label = ctk.CTkLabel(self, text="Sigma")
        self.gaussian_sigma_label.grid(row=5, column=0, sticky="w", padx=10)
        self.gaussian_sigma_slider = ctk.CTkSlider(self, from_=0.1, to=5.0, number_of_steps=50)
        self.gaussian_sigma_slider.set(1.0)
        self.gaussian_sigma_slider.grid(row=6, column=0, padx=10, pady=(0, 10))

        # Sharpen
        self.sharpen_filter_checkbox = ctk.CTkCheckBox(self, text="Sharpen")
        self.sharpen_filter_checkbox.grid(row=7, column=0, sticky="w", pady=5, padx=10)

        self.sharpen_strength_label = ctk.CTkLabel(self, text="Strength")
        self.sharpen_strength_label.grid(row=8, column=0, sticky="w", padx=10)
        self.sharpen_strength_slider = ctk.CTkSlider(self, from_=0.0, to=5.0, number_of_steps=50)
        self.sharpen_strength_slider.set(1.0)
        self.sharpen_strength_slider.grid(row=9, column=0, padx=10, pady=(0, 10))



class RightPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(row=0, column=2, rowspan=2, padx=20, pady=20, sticky="nsew")

        self.negative_checkbox = customtkinter.CTkCheckBox(self, text='negative')
        self.negative_checkbox.grid(row=0, column=0, sticky="nsew", pady=10, padx=10)

        self.black_and_white = customtkinter.CTkCheckBox(self, text='black and white')
        self.black_and_white.grid(row=1, column=0, sticky="nsew", pady=10, padx=10)

        self.binarize_checkbox = customtkinter.CTkCheckBox(self, text='binarize')
        self.binarize_checkbox.grid(row=2, column=0, sticky="nsew", pady=10, padx=10)
        self.threshold = ctk.CTkLabel(self, text='Binary threshold')
        self.threshold.grid(row=3, column=0, sticky="nsew")
        self.threshold = ctk.CTkSlider(self, from_=0, to=255, number_of_steps=255)
        self.threshold.set(10)
        self.threshold.grid(row=4, column=0, pady=10)

        self.add_noise = customtkinter.CTkLabel(self, text='Add noise')
        self.add_noise.grid(row=5, column=0, sticky="nsew", pady=10, padx=10)
        self.add_noise_slider = customtkinter.CTkSlider(self, from_=0, to=1, number_of_steps=20)
        self.add_noise_slider.set(0)
        self.add_noise_slider.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")


class MainPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.cached_roberts = None
        self.cached_sobel = None
        self.last_roberts_state = False
        self.last_sobel_state = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.side_panel = SidePanel(self)
        self.side_panel.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        self.Edges_Panel = Edges_Panel(self)
        self.Edges_Panel.grid(row=1, column=0, padx=10, pady=10, sticky="s")

        self.right_panel = RightPanel(self)
        self.right_panel.grid(row=0, column=2, sticky="n")

        self.filter_panel = FilterPanel(self)
        self.filter_panel.grid(row=1, column=2, sticky="s")

        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=1, pady=10, padx=10, sticky="ew")
        self.undo_btn = ctk.CTkButton(self.top_frame, text="Undo", width=60)
        self.undo_btn.pack(side="left", padx=5)
        self.redo_btn = ctk.CTkButton(self.top_frame, text="Redo", width=60, state='disabled')
        self.redo_btn.pack(side="left", padx=5)
        self.undo_stack = []
        self.redo_stack = []
        self.undo_btn.configure(command=self.undo)
        self.redo_btn.configure(command=self.redo)
        self.undo_btn.configure(state="disabled")
        self.redo_btn.configure(state="normal")

        # w tej ramce wyświetla się zdjęcie
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.select_btn = ctk.CTkButton(self.image_frame, text="Select image", command=self.load_image)
        self.select_btn.grid(row=2, column=1, pady=20, padx=20, sticky="nsew")

        #na tym rysuje się zdjęcie
        self.canvas = ctk.CTkCanvas(self.image_frame, bg="black", width=500, height=300)
        self.canvas.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

        # tutaj histogram
        self.histogram_frame = ctk.CTkFrame(self.image_frame)
        self.histogram_frame.grid(row=4, column=1, pady=10)

        self.button_save = ctk.CTkButton(self.image_frame, text="Save result", command=self.save_image)
        self.button_save.grid(row=5, column=1, pady=20)

        self.original_image_array = None
        self.processed_image_array = None

        self.side_panel.brightness_slider.configure(command=self.update_image)
        self.side_panel.contrast.configure(command=self.update_image)
        self.side_panel.contrast_slider.configure(command=self.update_image)
        self.right_panel.threshold.configure(command=self.update_image)
        self.right_panel.negative_checkbox.configure(command=self.update_image)
        self.right_panel.black_and_white.configure(command=self.update_image)
        self.right_panel.binarize_checkbox.configure(command=self.update_image)
        self.right_panel.add_noise_slider.configure(command=self.update_image)
        self.filter_panel.avg_filter_checkbox.configure(command=self.update_image)
        self.filter_panel.avg_kernel_slider.configure(command=self.update_image)
        self.filter_panel.sharpen_strength_slider.configure(command=self.update_image)
        self.filter_panel.sharpen_filter_checkbox.configure(command=self.update_image)
        self.filter_panel.gaussian_filter_checkbox.configure(command=self.update_image)
        self.filter_panel.gaussian_sigma_slider.configure(command=self.update_image)
        self.side_panel.vignette_slider.configure(command=self.update_image)
        self.Edges_Panel.roberts_cross_checkbox.configure(command=self.update_image)
        self.Edges_Panel.sobel_checkbox.configure(command=self.update_image)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.original_image_array= np.array(Image.open(file_path))
            self.processed_image_array = self.original_image_array.copy()
            self.display_image()
            self.display_histogram()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.undo_btn.configure(state="disabled")
        self.redo_btn.configure(state="disabled")

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.processed_image_array.copy())
            self.processed_image_array = self.undo_stack.pop()
            self.display_image()
            self.display_histogram()

            if not self.undo_stack:
                self.undo_btn.configure(state="disabled")
            self.redo_btn.configure(state="normal")

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.processed_image_array.copy())
            self.processed_image_array = self.redo_stack.pop()
            self.display_image()
            #self.display_histogram()

            if not self.redo_stack:
                self.redo_btn.configure(state="disabled")
            self.undo_btn.configure(state="normal")

    def update_image(self, event=None):
        if self.processed_image_array is not None:
            self.undo_stack.append(self.processed_image_array.copy())
            if len(self.undo_stack) > 5:
                self.undo_stack.pop(0)
            self.undo_btn.configure(state="normal")
            self.redo_stack.clear()
            self.redo_btn.configure(state="disabled")

        image = self.original_image_array.copy()


        brightness = self.side_panel.brightness_slider.get()
        image = self.adjust_brigthness(image, brightness)
        contrast = self.side_panel.contrast_slider.get()
        vignette = self.side_panel.vignette_slider.get()
        image = self.vignette(image, vignette)
        noise = self.right_panel.add_noise_slider.get()
        image= self.add_noise(image, noise)
        current_roberts_state = self.Edges_Panel.roberts_cross_checkbox.get()
        current_sobel_state = self.Edges_Panel.sobel_checkbox.get()

        if current_roberts_state != self.last_roberts_state:
            self.cached_roberts = None  # Wyczyść cache jeśli stan się zmienił
        self.last_roberts_state = current_roberts_state

        if current_sobel_state != self.last_sobel_state:
            self.cached_sobel = None
        self.last_sobel_state = current_sobel_state

        if self.Edges_Panel.roberts_cross_checkbox.get():
            image = self.roberts_cross(image)

        if self.Edges_Panel.sobel_checkbox.get():
            image = self.sobel_operator(image)

        if self.side_panel.contrast.get():
            image = self.adjust_contrast(image, contrast)

        if self.right_panel.negative_checkbox.get():
            image = self.negative(image)

        if self.right_panel.black_and_white.get():
            image = self.to_greyscale(image)

        if self.right_panel.binarize_checkbox.get():
            threshold = int(self.right_panel.threshold.get())
            image = self.binarize(image, threshold)

        if self.filter_panel.avg_filter_checkbox.get():
            size = int(self.filter_panel.avg_kernel_slider.get())
            if size % 2 == 0: size += 1
            image = self.apply_average_filter(image, kernel_size=size)

        if self.filter_panel.gaussian_filter_checkbox.get():
            sigma = self.filter_panel.gaussian_sigma_slider.get()
            size = 2 * int(3 * sigma) + 1
            image = self.apply_gaussian_filter(image, sigma=sigma, kernel_size=size)

        if self.filter_panel.sharpen_filter_checkbox.get():
            strength = self.filter_panel.sharpen_strength_slider.get()
            image = self.apply_sharpen_filter(image, strength=strength)

        self.processed_image_array = image
        self.display_image()

        if self.right_panel.binarize_checkbox.get():
            self.display_projection(self.processed_image_array)
        else:
            self.display_histogram()

    def adjust_brigthness(self, image, brightness):
        return np.clip(image + brightness, 0, 255).astype(np.uint8)

    def adjust_contrast(self, image, contrast):
        return 255*(image/np.max(image))**contrast

    def vignette(self, image, factor):
        height, width, _ = image.shape
        x = np.linspace(-1,1, width)
        y = np.linspace(-1,1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X ** 2 + Y ** 2)
        vignette_mask = np.exp(-factor*(distance**2))
        image = (image * vignette_mask[:, :, np.newaxis]).astype(np.uint8)
        return image


    def negative(self, image):
        return 255 - image

    def to_greyscale(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def binarize(self, image, threshold = 128):
        # binaryzacja
        if len(image.shape) == 3:
            image = self.to_greyscale(image)

        binary_image = np.where(image >= threshold, 255, 0).astype(np.uint8)

        if binary_image.ndim == 1:

            height = int(np.sqrt(binary_image.size))
            binary_image = binary_image.reshape((height, height))

        return binary_image

    def add_noise(self, image, noise_factor):
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        noisy_image = np.clip(image + noise * noise_factor, 0,255).astype(np.uint8)
        return noisy_image

    def convolve(self, image, kernel):
        if len(image.shape) == 3:
            return np.stack([self.convolve_channel(image[..., c], kernel) for c in range(3)], axis=-1)
        else:
            return self.convolve_channel(image, kernel)

    def convolve_channel(self, channel, kernel):
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(channel, ((ph, ph), (pw, pw)), mode='reflect')
        output = np.zeros_like(channel)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                output[i, j] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)

    def roberts_cross(self, image):
        if self.cached_roberts is not None and np.array_equal(image, self.cached_roberts['input']):
            return self.cached_roberts['output']
        kernel_x =np.array([[1,0],
                            [0,-1]])
        kernel_y = np.array([[0,1],
                             [-1,0]])

        grad_x = self.convolve(image, kernel_x)
        grad_y = self.convolve(image, kernel_y)

        gradient_maginitude = np.sqrt(grad_x**2+grad_y**2)
        result = np.clip(gradient_maginitude, 0, 255).astype(np.uint8)
        self.cached_roberts = {
            'input': image.copy(),
            'output': result.copy()
        }
        return result

    def sobel_operator(self, image):
        if len(image.shape) == 3:
            image = self.to_greyscale(image)

        # dla 0 stopni
        kernel_x = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])

        # dla 90 stopni
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])

        grad_x = self.convolve(image, kernel_x)
        grad_y = self.convolve(image, kernel_y)

        sobel = np.sqrt(grad_x.astype(np.float32) ** 2 + grad_y.astype(np.float32) ** 2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)

        self.cached_sobel = {
            'input': image.copy(),
            'output': sobel.copy()
        }

        return np.stack((sobel,) * 3, axis=-1)

    def apply_average_filter(self, image, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return self.convolve(image, kernel)

    def apply_gaussian_filter(self, image, sigma=1.0, kernel_size=5):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        gauss = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = np.outer(gauss, gauss)
        kernel /= np.sum(kernel)
        return self.convolve(image, kernel)

    def apply_sharpen_filter(self, image, strength=1.0):
        kernel = np.array([[0, -1, 0],
                           [-1, 4 + strength, -1],
                           [0, -1, 0]])
        return self.convolve(image, kernel)

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

    def display_projection(self, image):
        for widget in self.histogram_frame.winfo_children():
            widget.destroy()

        if len(image.shape) == 3:
            image = self.to_greyscale(image)

        binary_image = np.where(image >= 128, 1, 0)

        vertical_proj = np.sum(binary_image, axis=0)
        horizontal_proj = np.sum(binary_image, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3), gridspec_kw={'height_ratios': [1, 1]})
        ax1.plot(horizontal_proj)
        ax1.set_title("Horizontal projection")
        ax1.set_xlim(0, len(horizontal_proj))

        ax2.plot(vertical_proj)
        ax2.set_title("Vertical projection")
        ax2.set_xlim(0, len(vertical_proj))

        plt.tight_layout()

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
        # dodana funkcjonalność klawiszowa
        self.bind("<Control-z>", lambda event: self.main_panel.undo())
        self.bind("<Control-y>", lambda event: self.main_panel.redo())


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()