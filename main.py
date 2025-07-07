from utils import *
from trackers import *
from team_assigner import *
from player_ball_assigner import *
import numpy as np
from camera_movement_estimator import *
from speed_and_distance_estimator import *
from view_transformer import *
from pitch_visualization import *
from homography_transformer import *
from voronoi_diagram import *


def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')


    tracker = Tracker('models/best1.pt')
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pkl')
    
    #Znalezienie pozycji obiektu
    tracker.add_position_to_trakcs(tracks)
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_posistions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    homography_transformer = HomographyTransformer()
    
    # Dodaj pole 'position_homography' dla mapy
    for frame_num in range(len(tracks['field_keypoints'])):
        # Pobierz keypoints dla tej klatki
        if frame_num < len(tracks['field_keypoints']):
            field_keypoints_dict = tracks['field_keypoints'][frame_num]
            keypoints = field_keypoints_dict.get(1, {}).get('keypoints', [])
        else:
            keypoints = []
        
        # Oblicz homografię dla tej klatki
        homography_matrix = homography_transformer.calculate_homography(keypoints)
        
        # Transformuj pozycje graczy
        if frame_num < len(tracks['players']):
            for player_id, player in tracks['players'][frame_num].items():
                if 'position_adjusted' in player:
                    image_position = player['position_adjusted']
                    world_position = homography_transformer.transform_point(image_position, homography_matrix)
                    tracks['players'][frame_num][player_id]['position_homography'] = world_position
                else:
                    tracks['players'][frame_num][player_id]['position_homography'] = None
        
        # Transformuj pozycje sędziów
        if frame_num < len(tracks['referees']):
            for referee_id, referee in tracks['referees'][frame_num].items():
                if 'position_adjusted' in referee:
                    image_position = referee['position_adjusted']
                    world_position = homography_transformer.transform_point(image_position, homography_matrix)
                    tracks['referees'][frame_num][referee_id]['position_homography'] = world_position
                else:
                    tracks['referees'][frame_num][referee_id]['position_homography'] = None
        
        # Transformuj pozycję piłki
        if frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
            ball = tracks['ball'][frame_num][1]
            if 'position_adjusted' in ball:
                image_position = ball['position_adjusted']
                world_position = homography_transformer.transform_point(image_position, homography_matrix)
                tracks['ball'][frame_num][1]['position_homography'] = world_position
            else:
                tracks['ball'][frame_num][1]['position_homography'] = None
        
        if frame_num % 100 == 0:
            print(f"   📍 Homografia klatka {frame_num}/{len(tracks['field_keypoints'])}")
    
    print(f"✅ Transformacja homografii zakończona")
    
    # Statystyki obiektów na mapie
    if len(tracks['players']) > 0:
        avg_players = sum(len(frame_players) for frame_players in tracks['players']) / len(tracks['players'])
        print(f"   👥 Średnio graczy na mapie: {avg_players:.1f}")
    
    if len(tracks['referees']) > 0:
        avg_referees = sum(len(frame_refs) for frame_refs in tracks['referees']) / len(tracks['referees'])
        print(f"   🟡 Średnio sędziów na mapie: {avg_referees:.1f}")



    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #
    player_assigner = PlayerBallAssigner()
    team_ball_controll = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_controll.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_controll.append(team_ball_controll[-1])
    
    team_ball_controll = np.array(team_ball_controll)
    
    
    #Dodawanie rysunkow do outputu
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_controll)
    
    #Dodanie movementu camery
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    speed_and_distance_estimator.draw_speed_distance(output_video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_videos.avi')

    sample_frame = 0
    has_homography = False
    if (sample_frame < len(tracks['players']) and 
        len(tracks['players'][sample_frame]) > 0):
        
        first_player = next(iter(tracks['players'][sample_frame].values()))
        if 'position_homography' in first_player and first_player['position_homography'] is not None:
            has_homography = True
            print(f"✅ Homografia działa: {first_player['position_homography']}")
    
    if has_homography:
        # 🎯 GENERUJ MAPĘ BOISKA (2. video)
        field_map_generator = FieldMapGenerator(map_width=1200, map_height=800)
        field_map_frames = field_map_generator.generate_field_map_video(
            tracks, 
            team_ball_controll, 
            len(video_frames),
            position_field='position_homography',
            method_name="HOMOGRAPHY TRANSFORM"
        )
        
        save_video(field_map_frames, 'output_videos/field_map.avi')
        print("✅ Zapisano mapę boiska: output_videos/field_map.avi")
        
        # 🎯 GENERUJ DIAGRAM VORONOI (3. video)
        print("🎨 Generowanie diagramu Voronoi...")
        voronoi_generator = VoronoiDiagramGenerator(map_width=1200, map_height=800)
        voronoi_frames = voronoi_generator.generate_voronoi_video(
            tracks, 
            len(video_frames),
            opacity=0.6  # Przezroczystość jak na załączniku
        )
        
        save_video(voronoi_frames, 'output_videos/voronoi_diagram.avi')
        print("✅ Zapisano diagram Voronoi: output_videos/voronoi_diagram.avi")
        
        # Statystyki Voronoi
        total_team_1_on_field = 0
        total_team_2_on_field = 0
        valid_frames = 0
        
        for frame_num in range(len(tracks['players'])):
            team_1_count = 0
            team_2_count = 0
            
            for player in tracks['players'][frame_num].values():
                if 'position_homography' in player and player['position_homography'] is not None:
                    if voronoi_generator.is_on_field(player['position_homography']):
                        team = player.get('team', 0)
                        if team == 1:
                            team_1_count += 1
                        elif team == 2:
                            team_2_count += 1
            
            if team_1_count > 0 or team_2_count > 0:
                total_team_1_on_field += team_1_count
                total_team_2_on_field += team_2_count
                valid_frames += 1
        
        if valid_frames > 0:
            avg_team_1 = total_team_1_on_field / valid_frames
            avg_team_2 = total_team_2_on_field / valid_frames
            print(f"   🎯 Średnio graczy na boisku: Team1={avg_team_1:.1f}, Team2={avg_team_2:.1f}")
    
if __name__ == '__main__':
    main()