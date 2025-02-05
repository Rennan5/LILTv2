def convert_ocr_format(box):
    '''
    Converts the box in (x1, y1, x2, y2, x3, y3, x4, y4) format to (x1, y1, x2, y2, w, h) format (Util algorithm)
    
    Args:
        box (list): list of coordinates in (x1, y1, x2, y2, x3, y3, x4, y4) format
    
    Returns:
        conv_box (list): list of coordinates in (x1, y1, x2, y2, w, h) format
    '''
    
    x1 = min(box[0], box[2], box[4], box[6])
    y1 = min(box[1], box[3], box[5], box[7])
    x2 = max(box[0], box[2], box[4], box[6])
    y2 = max(box[1], box[3], box[5], box[7])
    w = x2 - x1
    h = y2 - y1
    conv_box = [x1, y1, x2, y2, w, h]
        
    return conv_box

def sort_boxes(box):
    # Passo 1: Calcular cx e cy para cada caixa
    Nb = len(box)
    cx = [(b[0] + b[2]) / 2 for b in box]
    cy = [(b[1] + b[3]) / 2 for b in box]
    delta = sum(b[5] for b in box) / (2 * Nb)

    # Passo 2: Obter sequência de índices Sy ordenados por cy
    Sy = sorted(range(Nb), key=lambda i: cy[i])

    # Passo 3: Inicializar variáveis auxiliares
    id = 0
    cy_ids = [0]

    # Passos 4 a 10: Agrupar índices baseados na diferença de cy
    for j in range(1, Nb):
        if abs(cy[Sy[j]] - cy[Sy[j - 1]]) < delta:
            cy_ids.append(id)
        else:
            id += 1
            cy_ids.append(id)

    # Passos 11 a 17: Reordenar dentro de cada grupo baseado em cx
    for j in range(0, max(cy_ids) + 1):
        # Encontrar índices de início e fim do grupo atual
        start = cy_ids.index(j)
        end = len(cy_ids) - 1 - cy_ids[::-1].index(j)

        # Obter sequência de índices Sx ordenados por cx dentro do grupo
        Sx = sorted(Sy[start:end + 1], key=lambda i: cx[i])

        # Atualizar Sy com a nova ordem dentro do grupo
        Sy[start:end + 1] = Sx

    # Passo 18: Reordenar box com base em Sy
    sorted_box = [box[i] for i in Sy]

    return sorted_box

if __name__ == '__main__':
    # (x1, y1, x2, y2, x3, y3, x4, y4)
    original_ocr = [[0, 0, 10, 0, 10, 5, 0, 5], [0, 0, 10, 0, 5, 10, 0, 5], [0, 0, 10, 0, 5, 10, 0, 5], [10, 0, 20, 0, 20, 10, 10, 10]]
    print('formato original: (x1, y1, x2, y2, x3, y3, x4, y4)')
    print(original_ocr)
    
    original_ocr = convert_ocr_format(original_ocr)
    print('formato convertido: (x1, y1, x2, y2, w, h)')
    print(original_ocr)
    
    sorted_ocr = sort_boxes(original_ocr)
    print('formato aranjado:')
    print(sorted_ocr)
