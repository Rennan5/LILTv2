#def convert_ocr_format(boxes):

def ags_algorithm(boxes):
    ''''
    Adaptative Gap-aware Sorting (AGS) function
    
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
    
    # computation of the parameters above
    for box in boxes:
        cx.append((box[0] + box[2]) / 2)
        cy.append((box[1] + box[3]) / 2)
        h_total += box[5]
        
    sig = h_total / (Nb * 2)
    