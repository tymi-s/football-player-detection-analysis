from ultralytics import YOLO
import supervision as sv
import pandas as pd
import pickle, os, sys, cv2
import numpy as np

sys.path.append('../')

from utils import *

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.field_model = None  # Model do keypoints
        self._init_field_model()

    
    def _init_field_model(self):
        """Inicjalizacja modelu keypoints boiska"""
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key="kb13wCgQ6mzGPAciRgVE")
            project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
            self.field_model = project.version(15).model
            print("✅ Model keypoints boiska załadowany")
        except Exception as e:
            print(f"⚠️ Nie udało się załadować modelu keypoints: {e}")
            self.field_model = None


    def _detect_field_keypoints(self, frame):
        """Wykryj keypoints boiska z jednej klatki"""
        if not self.field_model:
            return []
            
        try:
            # Zapisz tymczasowo
            temp_path = "temp_field_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Wykryj keypoints
            prediction = self.field_model.predict(temp_path)
            result = prediction.json()
            
            # Usuń temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result['predictions'] and result['predictions'][0]['predictions']:
                detection = result['predictions'][0]['predictions'][0]
                keypoints = detection['keypoints']
                confidence = detection['confidence']
                
                print(f"🔍 DEBUG - Keypoints z modelu:")
                for i, kp in enumerate(keypoints):
                    print(f"   {i}: class_id={kp['class_id']}, conf={kp['confidence']:.3f}, pos=({kp['x']}, {kp['y']})")
                
                unique_classes = set(kp['class_id'] for kp in keypoints)
                print(f"🔍 Unikalne class_id: {unique_classes}")
                
                # Filtruj keypoints z wysoką pewnością
                good_keypoints = []
                for kp in keypoints:
                    if kp['confidence'] > 0.4:
                        good_keypoints.append({
                            'x': int(kp['x']),
                            'y': int(kp['y']),
                            'confidence': kp['confidence'],
                            'class_id': kp['class_id']
                        })
                
                return good_keypoints
            else:
                print("❌ Nie wykryto keypoints boiska")
                return []
                
        except Exception as e:
            print(f"❌ Błąd wykrywania keypoints: {e}")
            return []
    

    def _create_keypoints_for_all_frames(self, frames, detect_every=20):
        """
        Wykryj keypoints co N klatek i przypisz do wszystkich klatek
        
        Args:
            frames: Lista wszystkich klatek
            detect_every: Co ile klatek wykrywać keypoints (20 = co 20 klatek)
        
        Returns:
            Lista keypoints dla każdej klatki
        """
        keypoints_cache = {}
        total_frames = len(frames)
        
        print(f"🎯 Wykrywanie keypoints co {detect_every} klatek z {total_frames} klatek...")
        
        # Wykryj keypoints co N klatek
        for frame_num in range(0, total_frames, detect_every):
            if frame_num % 100 == 0:
                print(f"   📍 Klatka {frame_num}/{total_frames}")
                
            keypoints = self._detect_field_keypoints(frames[frame_num])
            keypoints_cache[frame_num] = keypoints
        
        # Przypisz keypoints do wszystkich klatek
        all_keypoints = []
        for frame_num in range(total_frames):
            # Znajdź najbliższe wykryte keypoints
            best_detection_frame = min(keypoints_cache.keys(), 
                                     key=lambda x: abs(x - frame_num))
            keypoints = keypoints_cache[best_detection_frame]
            
            all_keypoints.append({1: {'keypoints': keypoints}})
        
        print(f"✅ Keypoints przypisane do {len(all_keypoints)} klatek")
        return all_keypoints
    
    def add_position_to_trakcs(self, tracks):
        for object, object_tracks in tracks.items():
            if object == 'field_keypoints':  
                continue
            for frame_num, track in enumerate(object_tracks):

                for track_id, track_info in track.items():
                
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_posistions):
        ball_posistions = [x.get(1, {}).get('bbox', []) for x in ball_posistions]
        df_ball_posistions = pd.DataFrame(ball_posistions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_posistions = df_ball_posistions.interpolate()
        df_ball_posistions  =df_ball_posistions.bfill()

        ball_posistions = [{1: {'bbox': x}} for x in df_ball_posistions.to_numpy().tolist()]

        return ball_posistions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.3)
            detections += detections_batch
            
        return detections
    
    def draw_field_keypoints(self, frame, keypoints):
        """Rysuj keypoints boiska na klatce - TYLKO PUNKTY"""
        if not keypoints:
            return frame
            
        good_points = 0
        for kp in keypoints:
            x, y = kp['x'], kp['y']
            conf = kp['confidence']
            class_id = kp['class_id']
            
            # Kolor zależny od pewności
            if conf > 0.8:
                color = (0, 255, 0)    # Zielony
            elif conf > 0.6:
                color = (0, 255, 255)  # Żółty
            else:
                color = (0, 165, 255)  # Pomarańczowy
            
            # Rysuj punkt
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
            
            # Dodaj numer class_id
            cv2.putText(frame, str(class_id), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            good_points += 1
        
        # Info o keypoints
        cv2.putText(frame, f"Field: {good_points} keypoints", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            # ============= SPRAWDŹ CZY TRACKS MA FIELD_KEYPOINTS =============
            if 'field_keypoints' not in tracks:
                print("⚠️ Stary stub bez keypoints - dodawanie keypoints...")
                tracks['field_keypoints'] = self._create_keypoints_for_all_frames(frames)
                
                # Zapisz zaktualizowany stub
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
                print("✅ Keypoints dodane do istniejącego stub'a")
            else:
                print(f"✅ Keypoints już w stub'ie: {len(tracks['field_keypoints'])} klatek")
            # ================================================================
            
            return tracks

        detections = self.detect_frames(frames)
        
        # ============= NOWA STRUKTURA Z KEYPOINTS =============
        tracks = {
            'players':[],
            'referees':[],
            'ball':[],
            'field_keypoints':[]  # ⭐ NOWE!
        }
        
        # ============= WYKRYJ KEYPOINTS Z PIERWSZEJ KLATKI =============
        print("🔍 Wykrywanie keypoints boiska...")
        tracks['field_keypoints'] = self._create_keypoints_for_all_frames(frames)
        # ==============================================================

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Zamiana bramkarza na piłkarza
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #Śledzenie obiektów
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}    
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}  

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_heigth = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2

        y1_rect = (y2 - rectangle_heigth//2) + 15
        y2_rect = (y2 + rectangle_heigth//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x - 10, y - 20],
            [x + 10, y -20]
        ]) 

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    
    def draw_team_controll(self, frame, frame_num, team_ball_controll):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_controll_till_frame = team_ball_controll[:frame_num+1]
        
        team_1_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame==1].shape[0]
        team_2_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame==2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f'Team1 Ball Controll: {100*team_1:.2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team2 Ball Controll: {100*team_2:.2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_controll):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # ============= RYSUJ KEYPOINTS BOISKA =============
            field_keypoints_dict = tracks['field_keypoints'][frame_num]
            if 1 in field_keypoints_dict:
                keypoints = field_keypoints_dict[1]['keypoints']
                frame = self.draw_field_keypoints(frame, keypoints)
            # ================================================

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))
            
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (255,0,0))
            
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))
            
            frame = self.draw_team_controll(frame, frame_num, team_ball_controll)
            output_video_frames.append(frame)

        return output_video_frames