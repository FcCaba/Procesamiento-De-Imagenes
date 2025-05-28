import httpx
from bs4 import BeautifulSoup

# URL de ejemplo (esto no devuelve resultados de imágenes reales)
url = 'https://www.google.com/search?q=objetos+geometricas&udm=2'

# Simular un navegador (user-agent)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

response = httpx.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Buscar todas las etiquetas <img>
imagenes = soup.find_all("img")
print(imagenes)
# Mostrar las URLs de las imágenes encontradas
for img in imagenes:
    src = img.get("src")
    if src:
        print(src)
