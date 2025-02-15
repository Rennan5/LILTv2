import numpy as np

class AGS:
    def __init__(self):
        pass

    def adaptive_gap_aware_sorting(self,bounding_boxes):
        """
        Implementa o Algoritmo de Ordenação Sensível a Lacunas (AGS) apenas considerando bounding boxes.
        
        Args:
            bounding_boxes: Lista de caixas delimitadoras [(x1, y1, x2, y2), ...].
        
        Returns:
            Lista ordenada de bounding boxes.
        """
        if not bounding_boxes:
            return []

        # Passo 1: Calcular a altura média dos blocos
        heights = [y2 - y1 for _, y1, _, y2 in bounding_boxes]
        avg_height = np.mean(heights)
        
        # Definir um limiar adaptativo para separação de linhas
        adaptive_gap = avg_height * 0.5  # Ajustável

        # Passo 2: Ordenar os bounding boxes pelo eixo Y
        sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][1])
        sorted_boxes = [bounding_boxes[i] for i in sorted_indices]

        # Passo 3: Agrupar por linhas baseado no gap adaptativo
        lines = []
        current_line = [sorted_boxes[0]]

        for i in range(1, len(sorted_boxes)):
            _, prev_y1, _, prev_y2 = sorted_boxes[i-1]
            _, curr_y1, _, curr_y2 = sorted_boxes[i]

            # Se a diferença entre as caixas for maior que o gap adaptativo, criar uma nova linha
            if curr_y1 - prev_y2 > adaptive_gap:
                lines.append(current_line)
                current_line = []

            current_line.append(sorted_boxes[i])
        
        if current_line:
            lines.append(current_line)

        # Passo 4: Ordenar cada linha pelo eixo X
        ordered_boxes = []
        for line in lines:
            line = sorted(line, key=lambda box: box[0])  # Ordenação horizontal (X)
            ordered_boxes.extend(line)

        return ordered_boxes