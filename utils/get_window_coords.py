import cv2

# Função de callback para capturar as coordenadas do clique
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Verifica se o botão esquerdo foi clicado
        print(f"Coordenadas do clique: x={x}, y={y}")

# Cria uma janela e associa o callback do mouse
cv2.namedWindow("Minha Janela", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Minha Janela", 500, 500)
cv2.setMouseCallback("Minha Janela", mouse_callback)

# Carrega e redimensiona uma imagem (ou um frame vazio)
image = cv2.imread('assets/images/naruto-pose.png')  # Substitua pelo caminho da sua imagem
if image is None:
    image = 255 * (image := cv2.imread('', 1))  # Cria uma imagem branca se não encontrar nenhuma
else:
    image = cv2.resize(image, (500, 500))  # Redimensiona a imagem para caber na janela

cv2.imshow("Minha Janela", image)

# Mantém a janela aberta até pressionar uma tecla
cv2.waitKey(0)
cv2.destroyAllWindows()