import cv2
import numpy as np

class HomographyTransformer:
    def __init__(self):
        # Współrzędne keypoints na prawdziwym boisku (w metrach)
        # Boisko FIFA: 105m x 68m, środek w (0, 0)
        # Współrzędne keypoints na prawdziwym boisku (w metrach)
        # Boisko FIFA: 105m x 68m, środek w (0, 0)
        # Na podstawie diagramu keypoints detection
        self.world_keypoints = {
            # LEWA STRONA BOISKA (ujemne X)
            0: np.array([-52.5, 34], dtype=np.float32),      # Lewy górny róg
            1: np.array([-52.5, 9.15], dtype=np.float32),   # Lewe pole bramkowe - góra
            2: np.array([-47, 9.15], dtype=np.float32),     # Lewe pole bramkowe - wewnętrzny róg góra
            3: np.array([-52.5, 5.5], dtype=np.float32),   # Lewa bramka - góra
            4: np.array([-52.5, -5.5], dtype=np.float32),  # Lewa bramka - dół
            5: np.array([-52.5, -34], dtype=np.float32),    # Lewy dolny róg
            6: np.array([-47, -9.15], dtype=np.float32),    # Lewe pole bramkowe - wewnętrzny róg dół
            7: np.array([-52.5, -9.15], dtype=np.float32), # Lewe pole bramkowe - dół
            8: np.array([-41, 0], dtype=np.float32),        # Punkt karny lewy
            9: np.array([-36.5, 20.15], dtype=np.float32), # Lewe pole karne - góra zewnętrzny
            10: np.array([-36.5, 9.15], dtype=np.float32), # Lewe pole karne - góra wewnętrzny
            11: np.array([-36.5, -9.15], dtype=np.float32), # Lewe pole karne - dół wewnętrzny
            12: np.array([-36.5, -20.15], dtype=np.float32), # Lewe pole karne - dół zewnętrzny
            
            # ŚRODEK BOISKA
            13: np.array([0, 34], dtype=np.float32),        # Górny punkt linii środkowej
            14: np.array([0, 9.15], dtype=np.float32),      # Górny punkt koła środkowego
            15: np.array([0, -9.15], dtype=np.float32),     # Dolny punkt koła środkowego
            16: np.array([0, -34], dtype=np.float32),       # Dolny punkt linii środkowej
            
            # PRAWA STRONA BOISKA (dodatnie X)
            17: np.array([36.5, 20.15], dtype=np.float32),  # Prawe pole karne - góra zewnętrzny
            18: np.array([36.5, 9.15], dtype=np.float32),   # Prawe pole karne - góra wewnętrzny
            19: np.array([41, 0], dtype=np.float32),         # Punkt karny prawy
            20: np.array([36.5, -9.15], dtype=np.float32),  # Prawe pole karne - dół wewnętrzny
            21: np.array([36.5, -20.15], dtype=np.float32), # Prawe pole karne - dół zewnętrzny
            22: np.array([47, 9.15], dtype=np.float32),     # Prawe pole bramkowe - wewnętrzny róg góra
            23: np.array([52.5, 9.15], dtype=np.float32),   # Prawe pole bramkowe - góra
            24: np.array([52.5, 34], dtype=np.float32),     # Prawy górny róg
            25: np.array([52.5, 5.5], dtype=np.float32),    # Prawa bramka - góra
            26: np.array([52.5, -5.5], dtype=np.float32),   # Prawa bramka - dół
            27: np.array([52.5, -9.15], dtype=np.float32),  # Prawe pole bramkowe - dół
            28: np.array([52.5, -34], dtype=np.float32),    # Prawy dolny róg
            29: np.array([47, -9.15], dtype=np.float32),    # Prawe pole bramkowe - wewnętrzny róg dół
            
            # KOŁO ŚRODKOWE
            30: np.array([-9.15, 0], dtype=np.float32),     # Lewy punkt koła środkowego
            31: np.array([9.15, 0], dtype=np.float32),      # Prawy punkt koła środkowego
        }
    
    def filter_keypoints(self, keypoints, confidence_threshold=0.5):
        """Filtruje keypoints po confidence (jak w Roboflow sports)"""
        if not keypoints:
            return []
        
        filtered = []
        for kp in keypoints:
            if kp['confidence'] > confidence_threshold:
                filtered.append(kp)
        
        return filtered
    
    def calculate_homography(self, keypoints):
        """
        Oblicza homografię na podstawie keypoints (jak w Roboflow sports)
        Wymaga minimum 4 punktów
        """
        # Filtruj keypoints
        good_keypoints = self.filter_keypoints(keypoints, confidence_threshold=0.4)  # Obniżony próg
        
        if len(good_keypoints) < 4:
            # print(f"⚠️ Za mało keypoints do homografii: {len(good_keypoints)} < 4")
            return None
        
        # Przygotuj punkty źródłowe (obraz) i docelowe (boisko)
        image_points = []
        world_points = []
        
        for kp in good_keypoints:
            class_id = kp['class_id']
            if class_id in self.world_keypoints:
                # Punkt na obrazie
                image_points.append([kp['x'], kp['y']])
                # Odpowiadający punkt na prawdziwym boisku
                world_points.append(self.world_keypoints[class_id])
        
        if len(image_points) < 4:
            # print(f"⚠️ Za mało znanych keypoints: {len(image_points)} < 4")
            # print(f"   Dostępne class_id: {[kp['class_id'] for kp in good_keypoints]}")
            # print(f"   Znane class_id: {list(self.world_keypoints.keys())}")
            return None
        
        # Konwertuj do numpy arrays
        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)
        
        # print(f"✅ Obliczam homografię z {len(image_points)} punktów")
        
        # Oblicz homografię (jak w Roboflow sports)
        try:
            homography_matrix, mask = cv2.findHomography(
                image_points, 
                world_points, 
                cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if homography_matrix is not None:
                # Sprawdź jakość homografii
                inliers = np.sum(mask) if mask is not None else len(image_points)
                # print(f"✅ Homografia obliczona: {inliers}/{len(image_points)} inliers")
                return homography_matrix
            else:
                # print("❌ Nie udało się obliczyć homografii")
                return None
                
        except Exception as e:
            # print(f"❌ Błąd obliczania homografii: {e}")
            return None
    
    def transform_point(self, point, homography_matrix):
        """Transformuje punkt z obrazu na prawdziwe boisko"""
        if homography_matrix is None or point is None:
            return None
        
        try:
            # Punkt na obrazie (x, y)
            image_point = np.array([[point]], dtype=np.float32)
            
            # Transformuj używając homografii
            world_point = cv2.perspectiveTransform(image_point, homography_matrix)
            
            # Wynik transformacji
            result = world_point[0][0].tolist()
            
            # ⭐ OPCJA: Odbij współrzędne jeśli boisko jest w lustrzanym odbiciu
            # Odkomentuj jedną z linii poniżej jeśli pozycje są odwrócone:
            
            # result[0] = -result[0]  # Odbij oś X (lewo-prawo)
            result[1] = -result[1]  # Odbij oś Y (góra-dół)
            # result = [result[1], result[0]]  # Zamień osie miejscami
            
            return result
            
        except Exception as e:
            # print(f"❌ Błąd transformacji punktu: {e}")
            return None
    
    def debug_keypoints_mapping(self, keypoints):
        """Debug - pokaż które keypoints mogą być zmapowane"""
        print("🔍 DEBUG - mapowanie keypoints:")
        available_classes = [kp['class_id'] for kp in keypoints]
        known_classes = list(self.world_keypoints.keys())
        
        print(f"   Dostępne class_id: {sorted(set(available_classes))}")
        print(f"   Znane w mapowaniu: {sorted(known_classes)}")
        
        mappable = [c for c in available_classes if c in known_classes]
        print(f"   Możliwe do zmapowania: {sorted(set(mappable))} ({len(set(mappable))} unikalnych)")
        
        if len(set(mappable)) < 4:
            print("   ⚠️ Za mało punktów - dodaj więcej do self.world_keypoints")
            
            # Sugestie nowych punktów
            unmapped = [c for c in set(available_classes) if c not in known_classes]
            if unmapped:
                print(f"   💡 Sugestia: dodaj te class_id do mapowania: {sorted(unmapped)}")
        
        return mappable