import flet as ft
import cv2
import numpy as np
import os
import base64
import asyncio
from datetime import datetime
from pathlib import Path

# Generar imagen en blanco base64 para preview
blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
_, blank_encoded = cv2.imencode(".png", blank_image)
BLANK_IMAGE_BASE64 = base64.b64encode(blank_encoded).decode()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CONFIG = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (100, 100),
    "captureDelay": 15,
    "minFaceWidth": 150,
    "brightnessThreshold": 50,
    "targetSize": (224, 224),
}


async def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.LIGHT
    page.title = "Capturador Facial"

    capturando = False
    cap = None
    contador_capturas = 0
    video_task = None
    tiempo_task = None
    capture_delay = 0
    tiempo_contador = 0
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    def show_snackbar(message):
        page.snack_bar = ft.SnackBar(ft.Text(message))
        page.snack_bar.open = True
        page.update()

    async def obtener_camaras():
        try:
            camaras.options = []
            for i in range(3):
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    camaras.options.append(ft.DropdownOption(text=f"Cámara {i}"))
                    temp_cap.release()
            page.update()
        except Exception as e:
            print(f"Error detectando cámaras: {e}")

    async def handle_camera_change(e):
        nonlocal cap, video_task
        if camaras.value:
            try:
                selected_cam = int(camaras.value.split()[-1])

                if cap is not None:
                    cap.release()
                    cap = None

                cap = cv2.VideoCapture(selected_cam)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                if video_task:
                    video_task.cancel()

                video_task = asyncio.create_task(actualizar_video())

            except Exception as e:
                print(f"Error cambiando cámara: {e}")

    async def actualizar_tiempo():
        nonlocal tiempo_contador
        while capturando:
            await asyncio.sleep(1)
            tiempo_contador += 1
            page.update()

    async def actualizar_video():
        nonlocal contador_capturas, capture_delay
        while cap and cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_processed = cv2.equalizeHist(gray)
                    gray_processed = cv2.GaussianBlur(gray_processed, (3, 3), 0)

                    faces = face_cascade.detectMultiScale(
                        gray_processed,
                        scaleFactor=CONFIG["scaleFactor"],
                        minNeighbors=CONFIG["minNeighbors"],
                        minSize=CONFIG["minSize"],
                    )

                    frame_visualizacion = frame.copy()
                    iluminacion = 0

                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        cv2.rectangle(
                            frame_visualizacion, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )

                        rostro_gris = gray_processed[y : y + h, x : x + w]
                        iluminacion = cv2.mean(rostro_gris)[0]

                        cv2.putText(
                            frame_visualizacion,
                            f"Iluminación: {iluminacion:.1f}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        bar_width = int(w * (iluminacion / 255))
                        cv2.rectangle(
                            frame_visualizacion,
                            (x, y + h + 10),
                            (x + bar_width, y + h + 20),
                            (0, 255, 0),
                            -1,
                        )

                    if capturando:
                        minutos = tiempo_contador // 60
                        segundos = tiempo_contador % 60
                        texto_tiempo = f"Tiempo: {minutos:02}:{segundos:02}"
                        texto_contador = (
                            f"Muestras: {contador_capturas}/{maxCapturasInput.value}"
                        )

                        cv2.putText(
                            frame_visualizacion,
                            texto_contador,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame_visualizacion,
                            texto_tiempo,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame_visualizacion,
                            f"{nombredataset.value} - #{contador_capturas}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    frame_principal = cv2.resize(frame_visualizacion, (640, 480))
                    _, img_principal = cv2.imencode(".png", frame_principal)
                    imageCamaraCaptura.src_base64 = base64.b64encode(
                        img_principal
                    ).decode()

                    if len(faces) > 0:
                        rostro = gray[y : y + h, x : x + w]
                        rostro = cv2.resize(rostro, (200, 200))
                        _, img_rostro = cv2.imencode(".png", rostro)
                        imageCamaraNormal.src_base64 = base64.b64encode(
                            img_rostro
                        ).decode()
                    else:
                        imageCamaraNormal.src_base64 = BLANK_IMAGE_BASE64

                    if capturando and len(faces) > 0:
                        if contador_capturas >= int(maxCapturasInput.value):
                            await cancelar_captura(None)
                        elif capture_delay <= 0 and w >= CONFIG["minFaceWidth"]:
                            if iluminacion >= CONFIG["brightnessThreshold"]:
                                try:
                                    rostro_guardar = cv2.resize(
                                        rostro_gris, CONFIG["targetSize"]
                                    )

                                    if not nombredataset.value.strip():
                                        raise ValueError(
                                            "El nombre del dataset es requerido"
                                        )

                                    if not rutatext.value or not os.path.exists(
                                        rutatext.value
                                    ):
                                        raise ValueError("Seleccione una ruta válida")

                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    nombre_base = nombredataset.value
                                    filename = f"{nombre_base}_{contador_capturas}_{timestamp}.png"

                                    listanomCapturas.controls.insert(
                                        0, ft.Text(filename, color=ft.Colors.WHITE)
                                    )
                                    listanomCapturas.controls = (
                                        listanomCapturas.controls[:10]
                                    )

                                    contador_capturas += 1
                                    capture_delay = CONFIG["captureDelay"]

                                    save_path = rutatext.value
                                    full_path = os.path.join(save_path, nombre_base)
                                    os.makedirs(full_path, exist_ok=True)
                                    cv2.imwrite(
                                        os.path.join(full_path, filename),
                                        rostro_guardar,
                                    )

                                except Exception as e:
                                    show_snackbar(str(e))
                                    await cancelar_captura(None)

                    capture_delay = max(0, capture_delay - 1)
                    page.update()

                await asyncio.sleep(0.03)
            except Exception as e:
                print(f"Error en video: {e}")
                break

    def get_default_dataset_path():
        downloads_path = str(Path.home() / "Downloads")
        dataset_path = os.path.join(downloads_path, "dataset")
        os.makedirs(dataset_path, exist_ok=True)
        return dataset_path

    async def iniciar_captura(e):
        nonlocal capturando, contador_capturas, tiempo_contador, tiempo_task, capture_delay
        # Validaciones
        if not nombredataset.value.strip():
            show_snackbar("Ingrese un nombre para el dataset")
            return

        if not rutatext.value or not os.path.exists(rutatext.value):
            show_snackbar("Seleccione una ruta válida")
            return

        try:
            int(maxCapturasInput.value)
        except:
            show_snackbar("Número de capturas inválido")
            return

        capturando = True
        contador_capturas = 0
        tiempo_contador = 0
        capture_delay = 0
        tiempo_task = asyncio.create_task(actualizar_tiempo())
        btnIniciar.disabled = True
        btnCancelar.disabled = False
        listanomCapturas.visible = True
        listanomCapturas.controls.clear()
        page.update()

    async def cancelar_captura(e):
        nonlocal capturando, contador_capturas, tiempo_contador, tiempo_task, capture_delay
        capturando = False
        if tiempo_task:
            tiempo_task.cancel()
        contador_capturas = 0
        tiempo_contador = 0
        capture_delay = 0
        nombredataset.value = ""
        listanomCapturas.controls.clear()
        btnIniciar.disabled = False
        btnCancelar.disabled = True
        listanomCapturas.visible = False
        page.update()

    async def seleccionar_directorio(e):
        try:
            def handle_result(result):
                if result.path:
                    # Autoagregar /dataset al path seleccionado
                    base_path = result.path
                    dataset_path = os.path.join(base_path, "dataset")
                    os.makedirs(dataset_path, exist_ok=True)
                    rutatext.value = dataset_path
                    page.update()
            
            file_picker.on_result = handle_result
            file_picker.get_directory_path()
            
        except Exception as e:
            print(f"Error seleccionando directorio: {e}")
            rutatext.value = ""
            page.update()

    maxCapturasInput = ft.TextField(
        label="Máximo de capturas",
        value="20",
        width=150,
        input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"),
    )

    camaras = ft.Dropdown(options=[], width=150, on_change=handle_camera_change)
    rutatext = ft.TextField(
        label="Ruta del dataset",
        expand=True,
        value=get_default_dataset_path(),  # Usar función de ruta predeterminada
    )

    btnbuscar = ft.IconButton(icon=ft.Icons.SEARCH, on_click=seleccionar_directorio)
    nombredataset = ft.TextField(label="Nombre del sujeto", width=200)
    btnIniciar = ft.Button(
        "Iniciar Captura", on_click=iniciar_captura, bgcolor=ft.Colors.GREEN_400
    )
    btnCancelar = ft.Button(
        "Cancelar", 
        on_click=cancelar_captura, 
        bgcolor=ft.Colors.RED_400,
        disabled=True
    )

    imageCamaraNormal = ft.Image(
        src_base64=BLANK_IMAGE_BASE64,
        width=200,
        height=200,
        fit=ft.ImageFit.CONTAIN,
        border_radius=10,
    )

    imageCamaraCaptura = ft.Image(fit=ft.ImageFit.CONTAIN)

    listanomCapturas = ft.Column(
        expand=True, scroll=ft.ScrollMode.ALWAYS, controls=[], spacing=5, visible=False
    )

    page.add(
        ft.Row(controls=[camaras, rutatext, btnbuscar]),
        ft.Row(controls=[nombredataset, maxCapturasInput, btnIniciar, btnCancelar]),
        ft.Divider(thickness=1),
        ft.Row(
            expand=True,
            controls=[
                ft.Container(
                    width=250,
                    content=ft.Column(
                        [
                            ft.Container(content=imageCamaraNormal, padding=10),
                            ft.Container(
                                expand=True,
                                content=ft.Column(
                                    [
                                        ft.Text(
                                            "Últimas capturas:",
                                            color=ft.Colors.WHITE,
                                            weight=ft.FontWeight.BOLD,
                                        ),
                                        listanomCapturas,
                                    ]
                                ),
                                bgcolor=ft.Colors.BLUE_600,
                                padding=10,
                                border_radius=10,
                            ),
                        ]
                    ),
                    bgcolor=ft.Colors.BLUE_400,
                    padding=10,
                ),
                ft.Container(
                    expand=True,
                    content=imageCamaraCaptura,
                    bgcolor=ft.Colors.BLUE_800,
                    padding=10,
                    border_radius=10,
                ),
            ],
        ),
    )

    
    await obtener_camaras()
    try:
        cap = cv2.VideoCapture(0)
        video_task = asyncio.create_task(actualizar_video())
    except Exception as e:
        print(f"Error inicializando cámara: {e}")

    async def on_close():
        nonlocal cap, video_task
        if cap and cap.isOpened():
            cap.release()
        if video_task:
            video_task.cancel()

    page.on_close = on_close


if __name__ == "__main__":
    ft.app(target=main)