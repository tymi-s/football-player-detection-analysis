import cv2
import numpy as np

class VoronoiDiagramGenerator:
    def __init__(self, map_width=1200, map_height=800):
        self.map_width = map_width
        self.map_height = map_height
        
        # Wymiary prawdziwego boiska (metry) - FIFA standard
        self.field_length = 105  
        self.field_width = 68
        
        # Skala: piksele na metr (z marginesami)
        self.scale_x = (self.map_width - 100) / self.field_length
        self.scale_y = (self.map_height - 100) / self.field_width
        
        # Offset dla centrowania
        self.offset_x = 50
        self.offset_y = 50
    
    def transform_position(self, world_pos):
        """Przekształca pozycję z metrów na piksele mapy"""
        if world_pos is None or len(world_pos) != 2:
            return None
            
        x_world, y_world = world_pos
        
        # Środek boiska = (0, 0) w metrach
        # Przekształć na piksele mapy
        x_map = int(self.offset_x + (x_world + self.field_length/2) * self.scale_x)
        y_map = int(self.offset_y + (y_world + self.field_width/2) * self.scale_y)
        
        # Ogranicz do granic mapy
        x_map = max(0, min(self.map_width-1, x_map))
        y_map = max(0, min(self.map_height-1, y_map))
        
        return (x_map, y_map)
    
    def create_field_background(self):
        """Tworzy tło boiska z liniami"""
        field = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        field[:] = (34, 139, 34)  # Kolor trawy
        
        line_color = (255, 255, 255)
        thickness = 3
        
        # Wymiary boiska w pikselach
        field_width_px = int(self.field_length * self.scale_x)
        field_height_px = int(self.field_width * self.scale_y)
        
        start_x, start_y = self.offset_x, self.offset_y
        end_x, end_y = start_x + field_width_px, start_y + field_height_px
        
        # Obrys boiska
        cv2.rectangle(field, (start_x, start_y), (end_x, end_y), line_color, thickness)
        
        # Linia środkowa
        center_x = start_x + field_width_px // 2
        cv2.line(field, (center_x, start_y), (center_x, end_y), line_color, thickness)
        
        # Koło środkowe
        center_y = start_y + field_height_px // 2
        radius = int(9.15 * self.scale_x)  # 9.15m promień
        cv2.circle(field, (center_x, center_y), radius, line_color, thickness)
        cv2.circle(field, (center_x, center_y), 2, line_color, -1)  # Punkt środkowy
        
        # Pola karne (16.5m x 40.3m)
        penalty_width = int(40.3 * self.scale_y)
        penalty_length = int(16.5 * self.scale_x)
        penalty_y_start = start_y + (field_height_px - penalty_width) // 2
        
        # Lewe pole karne
        cv2.rectangle(field, 
                     (start_x, penalty_y_start), 
                     (start_x + penalty_length, penalty_y_start + penalty_width), 
                     line_color, thickness)
        
        # Prawe pole karne
        cv2.rectangle(field, 
                     (end_x - penalty_length, penalty_y_start), 
                     (end_x, penalty_y_start + penalty_width), 
                     line_color, thickness)
        
        # Pola bramkowe (5.5m x 18.3m)
        goal_area_width = int(18.3 * self.scale_y)
        goal_area_length = int(5.5 * self.scale_x)
        goal_area_y_start = start_y + (field_height_px - goal_area_width) // 2
        
        # Lewe pole bramkowe
        cv2.rectangle(field, 
                     (start_x, goal_area_y_start), 
                     (start_x + goal_area_length, goal_area_y_start + goal_area_width), 
                     line_color, thickness)
        
        # Prawe pole bramkowe
        cv2.rectangle(field, 
                     (end_x - goal_area_length, goal_area_y_start), 
                     (end_x, goal_area_y_start + goal_area_width), 
                     line_color, thickness)
        
        # Bramki (7.32m szerokość)
        goal_width = int(7.32 * self.scale_y)
        goal_length = int(3 * self.scale_x)  # 3m głębokość
        goal_y_start = start_y + (field_height_px - goal_width) // 2
        
        # Lewa bramka
        cv2.rectangle(field, 
                     (start_x - goal_length, goal_y_start), 
                     (start_x, goal_y_start + goal_width), 
                     line_color, thickness)
        
        # Prawa bramka
        cv2.rectangle(field, 
                     (end_x, goal_y_start), 
                     (end_x + goal_length, goal_y_start + goal_width), 
                     line_color, thickness)
        
        # Punkty karne
        penalty_spot_distance = int(11 * self.scale_x)  # 11m od bramki
        penalty_spot_radius = 3
        
        # Lewy punkt karny
        left_penalty_x = start_x + penalty_spot_distance
        cv2.circle(field, (left_penalty_x, center_y), penalty_spot_radius, line_color, -1)
        
        # Prawy punkt karny
        right_penalty_x = end_x - penalty_spot_distance
        cv2.circle(field, (right_penalty_x, center_y), penalty_spot_radius, line_color, -1)
        
        return field
    
    def is_on_field(self, world_pos, margin=5):
        """Sprawdza czy pozycja jest na boisku"""
        if world_pos is None or len(world_pos) != 2:
            return False
        
        x, y = world_pos
        
        # Wymiary boiska FIFA: 105m x 68m, środek w (0,0)
        field_length_half = 52.5  # 105/2
        field_width_half = 34     # 68/2
        
        # Sprawdź czy w granicach + margines
        if (x >= -field_length_half - margin and x <= field_length_half + margin and
            y >= -field_width_half - margin and y <= field_width_half + margin):
            return True
        
        return False
    
    def calculate_voronoi_diagram(self, team_1_positions, team_2_positions, 
                                team_1_color=(128, 0, 128), team_2_color=(0, 128, 128)):
        """
        Oblicza diagram Voronoi na podstawie pozycji graczy
        """
        
        # Filtruj pozycje - tylko te na boisku
        valid_team_1 = [pos for pos in team_1_positions if self.is_on_field(pos)]
        valid_team_2 = [pos for pos in team_2_positions if self.is_on_field(pos)]
        
        if len(valid_team_1) == 0 and len(valid_team_2) == 0:
            return np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        
        # Stwórz diagramm Voronoi
        voronoi = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        
        # Konwertuj kolory na numpy arrays
        team_1_color = np.array(team_1_color, dtype=np.uint8)
        team_2_color = np.array(team_2_color, dtype=np.uint8)
        
        # Stwórz meshgrid dla wszystkich pikseli
        y_coords, x_coords = np.indices((self.map_height, self.map_width))
        
        # Konwertuj piksele na współrzędne świata (metry)
        world_x = (x_coords - self.offset_x) / self.scale_x - self.field_length/2
        world_y = (y_coords - self.offset_y) / self.scale_y - self.field_width/2
        
        # Funkcja do obliczania odległości
        def calculate_distances(positions, world_x, world_y):
            if len(positions) == 0:
                return np.full((self.map_height, self.map_width), np.inf)
            
            positions = np.array(positions)
            # Oblicz odległość od każdego piksela do każdego gracza
            distances = np.sqrt(
                (positions[:, 0][:, None, None] - world_x) ** 2 +
                (positions[:, 1][:, None, None] - world_y) ** 2
            )
            # Zwróć minimalną odległość dla każdego piksela
            return np.min(distances, axis=0)
        
        # Oblicz minimalne odległości dla każdej drużyny
        min_distances_team_1 = calculate_distances(valid_team_1, world_x, world_y)
        min_distances_team_2 = calculate_distances(valid_team_2, world_x, world_y)
        
        # Stwórz maskę kontroli - drużyna 1 kontroluje jeśli jest bliżej
        if len(valid_team_1) > 0 and len(valid_team_2) > 0:
            control_mask = min_distances_team_1 < min_distances_team_2
            voronoi[control_mask] = team_1_color
            voronoi[~control_mask] = team_2_color
        elif len(valid_team_1) > 0:
            # Tylko drużyna 1 na boisku
            valid_mask = min_distances_team_1 < np.inf
            voronoi[valid_mask] = team_1_color
        elif len(valid_team_2) > 0:
            # Tylko drużyna 2 na boisku
            valid_mask = min_distances_team_2 < np.inf
            voronoi[valid_mask] = team_2_color
        
        return voronoi
    
    def draw_players_on_voronoi(self, voronoi_field, team_1_positions, team_2_positions, team_colors):
        """Rysuje graczy na diagramie Voronoi"""
        
        # Rysuj graczy drużyny 1
        if 1 in team_colors:
            color = team_colors[1]
            for pos in team_1_positions:
                if self.is_on_field(pos):
                    map_pos = self.transform_position(pos)
                    if map_pos:
                        cv2.circle(voronoi_field, map_pos, 12, color, -1)
                        cv2.circle(voronoi_field, map_pos, 12, (255, 255, 255), 2)
        
        # Rysuj graczy drużyny 2
        if 2 in team_colors:
            color = team_colors[2]
            for pos in team_2_positions:
                if self.is_on_field(pos):
                    map_pos = self.transform_position(pos)
                    if map_pos:
                        cv2.circle(voronoi_field, map_pos, 12, color, -1)
                        cv2.circle(voronoi_field, map_pos, 12, (255, 255, 255), 2)
    
    def add_match_info_voronoi(self, voronoi_field, frame_num, total_frames, 
                             team_1_count, team_2_count, team_colors=None):
        """Dodaje informacje o meczu i kontroli przestrzeni"""
        info_height = 120
        cv2.rectangle(voronoi_field, (0, 0), (self.map_width, info_height), (0, 0, 0), -1)
        cv2.rectangle(voronoi_field, (0, 0), (self.map_width, info_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(voronoi_field, "VORONOI DIAGRAM - SPACE CONTROL", (20, 30), font, 0.8, (255, 255, 255), 2)
        
        # Czas meczu
        time_seconds = int((frame_num / total_frames) * 90 * 60)
        minutes, seconds = time_seconds // 60, time_seconds % 60
        cv2.putText(voronoi_field, f"Time: {minutes:02d}:{seconds:02d}", (20, 60), font, 0.6, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(voronoi_field, f"Frame: {frame_num+1}/{total_frames}", 
                   (self.map_width - 200, 30), font, 0.5, (255, 255, 255), 2)
        
        # Statystyki graczy na boisku
        cv2.putText(voronoi_field, f"Players on field: Team1={team_1_count}, Team2={team_2_count}", 
                   (20, 90), font, 0.5, (255, 255, 255), 2)
        
        # Legenda kolorów drużyn
        if team_colors:
            legend_x = self.map_width - 300
            cv2.putText(voronoi_field, "Teams:", (legend_x, 55), font, 0.5, (255, 255, 255), 2)
            
            y_offset = 15
            for team_id, color in team_colors.items():
                color_bgr = tuple(int(c) for c in color) if isinstance(color, (list, tuple)) else color
                cv2.circle(voronoi_field, (legend_x + 15, 55 + y_offset), 10, color_bgr, -1)
                cv2.circle(voronoi_field, (legend_x + 15, 55 + y_offset), 10, (255, 255, 255), 2)
                cv2.putText(voronoi_field, f"Team {team_id}", (legend_x + 30, 61 + y_offset), font, 0.4, (255, 255, 255), 1)
                y_offset += 20
    
    def generate_voronoi_video(self, tracks, total_frames, opacity=0.6):
        """
        Generuje video z diagramem Voronoi
        """
        voronoi_frames = []
        
        # Pobierz kolory drużyn z pierwszej klatki
        team_colors = {}
        if len(tracks['players']) > 0:
            for player in tracks['players'][0].values():
                if 'team' in player and 'team_color' in player:
                    team_colors[player['team']] = player['team_color']
        
        # Użyj rzeczywistych kolorów koszulek drużyn dla diagramu Voronoi
        voronoi_team_1_color = team_colors.get(1, (128, 0, 128))    # Kolor drużyny 1 lub domyślny różowy
        voronoi_team_2_color = team_colors.get(2, (0, 128, 128))   # Kolor drużyny 2 lub domyślny zielony
        
        for frame_num in range(total_frames):
            # Stwórz tło boiska
            field_background = self.create_field_background()
            
            # Zbierz pozycje graczy dla każdej drużyny
            team_1_positions = []
            team_2_positions = []
            
            if frame_num < len(tracks['players']):
                for track_id, player in tracks['players'][frame_num].items():
                    if 'position_homography' in player and player['position_homography'] is not None:
                        world_pos = player['position_homography']
                        team = player.get('team', 0)
                        
                        if team == 1:
                            team_1_positions.append(world_pos)
                        elif team == 2:
                            team_2_positions.append(world_pos)
            
            # Oblicz diagram Voronoi
            voronoi_diagram = self.calculate_voronoi_diagram(
                team_1_positions, 
                team_2_positions,
                voronoi_team_1_color,
                voronoi_team_2_color
            )
            
            # Nałóż diagram Voronoi na tło boiska z przezroczystością
            voronoi_field = cv2.addWeighted(voronoi_diagram, opacity, field_background, 1 - opacity, 0)
            
            # Rysuj graczy na diagramie
            self.draw_players_on_voronoi(voronoi_field, team_1_positions, team_2_positions, team_colors)
            
            # Dodaj informacje o meczu
            self.add_match_info_voronoi(voronoi_field, frame_num, total_frames, 
                                      len(team_1_positions), len(team_2_positions), team_colors)
            
            voronoi_frames.append(voronoi_field)
        
        return voronoi_frames