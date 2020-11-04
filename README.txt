YOLO v3 Player Detection and Classification Script

optional arguments:
  -h, --help            show this help message and exit
Required args:
  --settings_path 	Settings file path
  --sfm              Spatial filter method
  --bbm	             Bounding box method

Extra args:
  --video_path  	Location of video file
  --save_video           Save resulting video with bbox
  --split 	         Frame Splits
  --ce                   Contrast Enhancement
  --alpha 	         Contrast param (1.0 - 3.0)
  --beta                 Brightness param (1-100)
  --show_live 		 Watch live detections


Run:
(env) D:\>python detect_players.py --settings_path ./detect_players_settings.ini --bbm --sfm

Outputs:
Player.txt ( all player detections)
Parms.json ( all necesary params for tracking, include path to base directory, video frame height and width)
/img	   ( save video in images)
