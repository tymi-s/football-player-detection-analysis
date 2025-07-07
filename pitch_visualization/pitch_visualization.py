import cv2
import numpy as np

class FieldMapGenerator:
    def __init__(self, map_width=800, map_height=500):
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
        """
        Sprawdza czy pozycja jest na boisku (w granicach FIFA + margines)
        
        Args:
            world_pos: pozycja w metrach (x, y)
            margin: margines w metrach (domyślnie 5m)
        
        Returns:
            bool: True jeśli na boisku
        """
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
    
    def draw_player(self, field_map, position, color, player_id=None, has_ball=False):
        if position is None:
            return
            
        x, y = position
        radius = 15  # Większy rozmiar dla lepszej widoczności
        
        # Gracz - tylko kolorowa kropka (bez śladów i numerów)
        cv2.circle(field_map, (x, y), radius, color, -1)
        cv2.circle(field_map, (x, y), radius, (255, 255, 255), 2)  # Biała obwódka
        
        # Gracz z piłką - dodatkowa żółta obwódka
        if has_ball:
            cv2.circle(field_map, (x, y), radius + 5, (0, 255, 255), 3)
    
    def draw_ball(self, field_map, position):
        if position is None:
            return
        x, y = position
        radius = 10  # Większy rozmiar piłki
        
        # Piłka - biała kropka z czarną obwódką (bez śladu)
        cv2.circle(field_map, (x, y), radius, (255, 255, 255), -1)
        cv2.circle(field_map, (x, y), radius, (0, 0, 0), 2)
    
    def add_match_info(self, field_map, frame_num, total_frames, team_ball_control, method_name="FIELD MAP", team_colors=None):
        """Dodaje informacje o meczu na górze mapy"""
        info_height = 120
        cv2.rectangle(field_map, (0, 0), (self.map_width, info_height), (0, 0, 0), -1)
        cv2.rectangle(field_map, (0, 0), (self.map_width, info_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(field_map, f"FIELD MAP - {method_name}", (20, 30), font, 0.8, (255, 255, 255), 2)
        
        # Czas meczu
        time_seconds = int((frame_num / total_frames) * 90 * 60)
        minutes, seconds = time_seconds // 60, time_seconds % 60
        cv2.putText(field_map, f"Time: {minutes:02d}:{seconds:02d}", (20, 60), font, 0.6, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(field_map, f"Frame: {frame_num+1}/{total_frames}", 
                   (self.map_width - 200, 30), font, 0.5, (255, 255, 255), 2)
        
        # Legenda kolorów drużyn z rzeczywistymi kolorami
        if team_colors:
            legend_x = self.map_width - 300
            cv2.putText(field_map, "Legend:", (legend_x, 55), font, 0.5, (255, 255, 255), 2)
            
            y_offset = 15
            
            # Drużyny
            for team_id, color in team_colors.items():
                # Konwertuj kolor BGR na format OpenCV
                color_bgr = tuple(int(c) for c in color) if isinstance(color, (list, tuple)) else color
                cv2.circle(field_map, (legend_x + 15, 55 + y_offset), 10, color_bgr, -1)
                cv2.circle(field_map, (legend_x + 15, 55 + y_offset), 10, (255, 255, 255), 2)
                cv2.putText(field_map, f"Team {team_id}", (legend_x + 30, 61 + y_offset), font, 0.4, (255, 255, 255), 1)
                y_offset += 20
            
            # Sędziowie - żółty kolor (tylko ci na boisku)
            cv2.circle(field_map, (legend_x + 15, 55 + y_offset), 10, (0, 255, 255), -1)  # Żółty
            cv2.circle(field_map, (legend_x + 15, 55 + y_offset), 10, (255, 255, 255), 2)
            cv2.putText(field_map, "Referees*", (legend_x + 30, 61 + y_offset), font, 0.4, (255, 255, 255), 1)
            y_offset += 15
            
            # Notatka o filtrze
            cv2.putText(field_map, "*on field only", (legend_x + 30, 55 + y_offset), font, 0.3, (200, 200, 200), 1)
    
    def generate_field_map_video(self, tracks, team_ball_control, total_frames, 
                                position_field='position_transformed', method_name="STANDARD"):
        """
        Generuje video z mapą boiska
        
        Args:
            tracks: dane tracking
            team_ball_control: kontrola piłki przez drużyny
            total_frames: liczba klatek
            position_field: które pole pozycji używać ('position_transformed' lub 'position_homography')
            method_name: nazwa metody do wyświetlenia
        """
        field_map_frames = []
        
        # Pobierz kolory drużyn z pierwszej klatki
        team_colors = {}
        if len(tracks['players']) > 0:
            for player in tracks['players'][0].values():
                if 'team' in player and 'team_color' in player:
                    team_colors[player['team']] = player['team_color']
        
        # Kolor sędziów - żółty
        referee_color = (0, 255, 255)  # Żółty w BGR
        
        for frame_num in range(total_frames):
            field_map = self.create_field_background()
            self.add_match_info(field_map, frame_num, total_frames, team_ball_control, method_name, team_colors)
            
            # Gracze
            if frame_num < len(tracks['players']):
                for track_id, player in tracks['players'][frame_num].items():
                    if position_field in player:
                        world_pos = player[position_field]
                        map_pos = self.transform_position(world_pos)
                        
                        color = player.get('team_color', (0, 0, 255))
                        has_ball = player.get('has_ball', False)
                        
                        self.draw_player(field_map, map_pos, color, track_id, has_ball)
            
            # Sędziowie - żółty kolor (TYLKO CI NA BOISKU)
            if frame_num < len(tracks['referees']):
                for track_id, referee in tracks['referees'][frame_num].items():
                    if position_field in referee:
                        world_pos = referee[position_field]
                        
                        # ⭐ FILTR: Sprawdź czy sędzia jest na boisku
                        if self.is_on_field(world_pos, margin=3):  # 3m margines
                            map_pos = self.transform_position(world_pos)
                            
                            # Sędziowie są żółci i nigdy nie mają piłki
                            self.draw_player(field_map, map_pos, referee_color, track_id, has_ball=False)
            
            # Piłka
            if frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
                ball = tracks['ball'][frame_num][1]
                if position_field in ball:
                    world_pos = ball[position_field]
                    map_pos = self.transform_position(world_pos)
                    self.draw_ball(field_map, map_pos)
            
            field_map_frames.append(field_map)
        
        return field_map_frames