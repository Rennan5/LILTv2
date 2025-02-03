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


def ags_algorithm(boxes):
    ''''
    Adaptative Gap-aware Sorting (AGS) function (Algorithm 1 from the article)
    
    Args:
        boxes (list): list of boxes in (box_0, box_1, ..., box_i) format,
                      where box_i is for box_i = [x_1, y_1, x_2, y_2, w, h],
                      where (x_1, y_1) is top-left corner, (x_2, y_2) is bottom-right corner
                      and w is width and h is height.
        
    Returns:
        sorted_boxes (list): sorted boxes in (box_0, box_1, ..., box_i) format
    '''

    # declaration of the center of each box, number of boxes and sum of heights
    cx = []
    cy = []
    Nb = len(boxes)
    h_total = 0
    
    # computation of the parameters above (line 1 from the article)
    for box in boxes:
        cx.append((box[0] + box[2]) / 2)
        cy.append((box[1] + box[3]) / 2)
        h_total += box[5]    
    sig = h_total / (Nb * 2)
    
    # inicialization of indexes the smallest to the largest value in cy (line 2 from the article)
    Sy = [i for i in range(Nb)]
    Sy.sort(key=lambda x: cy[x])
    
    # declaration of the ids of the boxes and its cy (line 3 from the article)
    id = 0
    cy_ids = []
    
    # lines 4 to 11 from the article
    for j in range(1, Nb - 1):
        if cy[Sy[j]] - cy[Sy[j - 1]] < sig: cy_ids.append(id)
        else:
            id = id + 1
            cy_ids.append(id)
    
    # lines 12 to 17 from the article
    for j in range(0, max(cy_ids)):
        for i in len(cy_ids):
            if cy_ids[i] == j:
                start = i
                break
        for i in range(len(cy_ids), -1, -1):
            if cy_ids[i] == j:
                end = i
                break
            
        Sx = [i for i in range(len(cx[Sy[start:end]]))] # possivel erro (linha 15)
        Sx.sort(key=lambda x: cx[Sy[start:end]][x]) # possivel erro (linha 15)
        
        Sy[start:end] = [Sy[start:end][i] for i in Sx] # possivel erro (linha 16)
        
    # reordering the elements of boxes using the sequence of indices Sy (line 18 from the article)
    sorted_boxes = [boxes[Sy[i]] for i in range(Nb)] # possivel erro (linha 18)
    
    return sorted_boxes

if __name__ == '__main__':
    # (x1, y1, x2, y2, x3, y3, x4, y4)
    original_ocr = [[0, 0, 10, 0, 10, 5, 0, 5], [0, 0, 10, 0, 5, 10, 0, 5], [0, 0, 10, 0, 5, 10, 0, 5], [10, 0, 20, 0, 20, 10, 10, 10]]
    print('formato original: (x1, y1, x2, y2, x3, y3, x4, y4)')
    print(original_ocr)
    
    original_ocr = convert_ocr_format(original_ocr)
    print('formato convertido: (x1, y1, x2, y2, w, h)')
    print(original_ocr)
    
    sorted_ocr = ags_algorithm(original_ocr)
    print('formato aranjado:')
    print(sorted_ocr)
