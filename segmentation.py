import cv2
import mediapipe as mp
import numpy as np

# Inicializando o MediaPipe para segmentação de selfie
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Função para carregar o fundo da imagem
def load_background_image(background_path, frame_shape):
    # Carregar a imagem do fundo
    background = cv2.imread(background_path)
    
    if background is None:
        print("Erro: Não foi possível carregar a imagem de fundo. Verifique o caminho.")
        return None
    
    # Redimensionar o fundo para cobrir toda a tela (respeitando a altura e largura do frame)
    background_resized = cv2.resize(background, (frame_shape[1], frame_shape[0]))
    
    return background_resized

# Função para extrair a pessoa e criar clones
def extract_person_from_frame(frame, contour_coords):
    # Criando uma máscara binária da pessoa
    mask_person = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    
    for contour in contour_coords:
        # Criando uma máscara para o contorno
        cv2.drawContours(mask_person, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Usando a máscara para copiar a pessoa da imagem original
    person_region = cv2.bitwise_and(frame, frame, mask=mask_person)
    return person_region, mask_person

# Caminho do fundo comum
common_background_path = "assets/images/background.png"

# Captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertendo o frame para RGB (necessário para o MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processando a segmentação de fundo
    results = segmenter.process(frame_rgb)
    
    # Máscara binária da segmentação (pessoa vs fundo)
    mask = results.segmentation_mask
    
    # Criando a máscara binária com um limiar de 0.5 (só a pessoa)
    mask = mask > 0.5  # 1 para a pessoa e 0 para o fundo
    
    # Carregando e redimensionando o fundo comum
    common_background = load_background_image(common_background_path, frame.shape)

    # Verificar se o fundo foi carregado corretamente
    if common_background is None:
        print('Erro ao carregar o fundo comum.')
        continue

    # Encontrar os contornos da máscara binária (onde há branco, ou seja, a pessoa)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializando uma lista para armazenar as coordenadas dos contornos
    contour_coords = []

    # Para cada contorno encontrado, armazenamos as coordenadas
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filtra contornos pequenos
            contour_coords.append(contour)  # Guardando as coordenadas dos contornos

    # Extraindo a pessoa do frame (máscara da pessoa)
    person_region, person_mask = extract_person_from_frame(frame, contour_coords)

    # Dividindo a largura do frame em 3 partes
    clone_width = frame.shape[1] // 3
    clone_height = frame.shape[0] 

    # Garantindo que a largura total seja coberta (evitando diferenças de pixels)
    common_background_sections = [
        common_background[:, 0:clone_width],                # Primeira seção do fundo
        common_background[:, clone_width:2*clone_width],   # Segunda seção do fundo
        common_background[:, 2*clone_width:frame.shape[1]] # Terceira seção do fundo (cobre o resto)
    ]

    # Criando as três partes da tela com a pessoa clonada sobre os fundos
    final_frame = np.zeros_like(frame)

    for i in range(3):
        # Calculando onde colocar a ROI (pessoa) na tela dividida
        x_start = i * clone_width
        x_end = (i + 1) * clone_width if i < 2 else frame.shape[1]  # Última seção cobre o restante

        # Redimensionando a pessoa para caber na largura de cada parte da tela
        clone_section = cv2.resize(person_region, (x_end - x_start, clone_height))

        # Misturando a pessoa com a respectiva seção do fundo comum
        blended_section = np.where(
            clone_section > 0,  # Onde há pessoa
            clone_section,      # Mostra a pessoa
            common_background_sections[i]  # Mostra o fundo
        )

        # Colocando a seção mesclada no frame final
        final_frame[:, x_start:x_end] = blended_section

    # Exibindo o frame final com os clones e fundo ajustados
    cv2.imshow("Clones in 3 Sections with Common Background", final_frame)

    # Fechar o programa ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
