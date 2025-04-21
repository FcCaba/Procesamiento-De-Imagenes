import flet as ft
import cv2
import numpy as np
from PIL import Image
import base64


# Función para convertir imagen (cv2) a base64
def img_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


# App principal
def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.LIGHT
    page.title = "Detector de Bordes Interactivo"
    page.scroll = "auto"
    page.window_width = 1200

    # Variables de estado
    img = None
    sigma = 1.0
    laplacian_factor = 1.0
    log_factor = 1.0
    amplificar = False
    mostrar_color = False
    ancho_imagen = 300

    # Widgets imagen
    original_img = ft.Image(width=ancho_imagen, src=ft.icons.IMAGE)
    lap_img = ft.Image(width=ancho_imagen, src=ft.icons.IMAGE)
    log_img = ft.Image(width=ancho_imagen, src=ft.icons.IMAGE)

    titles = ft.Row(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[
            ft.Column(
                horizontal_alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Text(
                        "Original",
                        size=16,
                        weight="bold",
                        width=ancho_imagen,
                        text_align="center",
                    ),
                    original_img,
                ],
            ),
            ft.Column(
                horizontal_alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Text(
                        "Laplaciano",
                        size=16,
                        weight="bold",
                        width=ancho_imagen,
                        text_align="center",
                    ),
                    lap_img,
                ],
            ),
            ft.Column(
                horizontal_alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Text(
                        "LoG",
                        size=16,
                        weight="bold",
                        width=ancho_imagen,
                        text_align="center",
                    ),
                    log_img,
                ],
            ),
        ],
    )

    # Función de procesamiento
    def actualizar_imagenes():
        if img is None:
            return

        original = img if mostrar_color else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplaciano
        lap = cv2.Laplacian(original, cv2.CV_64F)
        lap *= laplacian_factor
        if amplificar:
            lap *= 1.5
        lap = cv2.convertScaleAbs(lap)

        # LoG
        blurred = cv2.GaussianBlur(original, (5, 5), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        log *= log_factor
        if amplificar:
            log *= 1.5
        log = cv2.convertScaleAbs(log)

        # Convertir imágenes a color si es necesario
        def format_img(im):
            return (
                cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                if len(im.shape) == 2 and mostrar_color
                else im
            )

        original_img.src_base64 = img_to_base64(format_img(original))
        lap_img.src_base64 = img_to_base64(format_img(lap))
        log_img.src_base64 = img_to_base64(format_img(log))

        original_img.width = lap_img.width = log_img.width = ancho_imagen

        original_img.update()
        lap_img.update()
        log_img.update()

    # Eventos
    def on_sigma_change(e):
        nonlocal sigma
        sigma = float(e.control.value)
        actualizar_imagenes()

    def on_lap_factor_change(e):
        nonlocal laplacian_factor
        laplacian_factor = float(e.control.value)
        actualizar_imagenes()

    def on_log_factor_change(e):
        nonlocal log_factor
        log_factor = float(e.control.value)
        actualizar_imagenes()

    def on_check_amplificar(e):
        nonlocal amplificar
        amplificar = e.control.value
        actualizar_imagenes()

    def on_check_color(e):
        nonlocal mostrar_color
        mostrar_color = e.control.value
        actualizar_imagenes()

    def on_image_size_change(e):
        nonlocal ancho_imagen
        ancho_imagen = int(e.control.value)
        actualizar_imagenes()

    # --- Cargar imagen con FilePicker ---
    file_picker = ft.FilePicker()

    def on_file_selected(e: ft.FilePickerResultEvent):
        nonlocal img
        if e.files:
            path = e.files[0].path
            img = cv2.imread(path)
            if img is not None:
                actualizar_imagenes()

    file_picker.on_result = on_file_selected

    # Botón para abrir selector
    abrir_btn = ft.ElevatedButton(
        "Seleccionar imagen",
        on_click=lambda _: file_picker.pick_files(allow_multiple=False),
    )

    # Sliders y checks
    sigma_slider = ft.Slider(
        min=0.5,
        max=5.0,
        value=sigma,
        divisions=20,
        label="Sigma (LoG): {value}",
        on_change=on_sigma_change,
    )
    lap_slider = ft.Slider(
        min=0.5,
        max=5.0,
        value=laplacian_factor,
        divisions=20,
        label="Intensidad Laplaciano: {value}",
        on_change=on_lap_factor_change,
    )
    log_slider = ft.Slider(
        min=0.5,
        max=5.0,
        value=log_factor,
        divisions=20,
        label="Intensidad LoG: {value}",
        on_change=on_log_factor_change,
    )
    color_checkbox = ft.Checkbox(
        label="Mostrar en color", value=False, on_change=on_check_color
    )
    amplificar_checkbox = ft.Checkbox(
        label="Aplicar más intensidad", value=False, on_change=on_check_amplificar
    )
    tamaño_slider = ft.Slider(
        min=100,
        max=600,
        value=ancho_imagen,
        divisions=10,
        label="Tamaño de imágenes: {value}",
        on_change=on_image_size_change,
    )

    # Agregar componentes
    page.overlay.append(file_picker)

    page.add(
        ft.Text(
            "Detector de Bordes Interactivo (Laplaciano y LoG)", size=22, weight="bold"
        ),
        abrir_btn,
        titles,
        ft.Divider(),
        sigma_slider,
        lap_slider,
        log_slider,
        tamaño_slider,
        ft.Row([color_checkbox, amplificar_checkbox]),
    )


if __name__ == "__main__":
    ft.app(target=main)
