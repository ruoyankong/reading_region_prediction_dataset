# Getting the Most from Eye-Tracking: User-Interaction Based Reading Region Estimation Dataset and Models

This is the dataset and code for _Ruoyan Kong, Ruixuan Sun, Chuankai Zhang, Chen Chen, Sneha Patr, Gayathri Gajjela, and Joseph A. Konstan.
Getting the Most from Eye-Tracking: User-Interaction Based Reading Region Estimation Dataset and Models. Symposium on Eye Tracking Research and Applications (ETRA 2023)._
               
## Data
### second_feature_labeled.csv (199682 rows)

Each row represents one participant's interaction with each message (story) per second and whether they are reading that message.

#### Info columns:
"userId", participants' ID (we have 9 participants)

"postId", which posts the participant is reading (each post has multiple messages)

"time_stamp", the number of seconds since the participant started reading this post (start from 0)

"index", the rank of the message in the post (start from 0)

"n_word", the number of words in the message

#### X (the participant's interactions with the message until the timestamp):
"MMF_y_2", "MMF_y_5", "MMF_y_10", "MMF_y_inf": user's mouse moving length in the y direction in the past 2/5/10/inf(since starting reading this post) seconds -- rescaled to 0 to 1 by screen height

"MMF_x_2","MMF_x_5","MMF_x_10","MMF_x_inf": user's mouse moving length in the x direction in the past 2/5/10/inf(since starting reading this post) seconds -- rescaled to 0 to 1 by screen width

"MSF_y_2","MSF_y_5","MSF_y_10","MSF_y_inf": user's mouse scrolling length in the y direction in the past 2/5/10/inf(since starting reading this post) seconds -- rescaled to 0 to 1 by screen height

"MSF_clk": the percentage of messages in this post that have been clicked by this user until this timestamp (0 to 1)

"isVisible": whether the message is visible on screen in the past second (True or False)

"mouseX","mouseY": the user's X/Y mouse position (0 to 1, rescaled by screen width / height)

"M_tclk": if at least one of the messages in the post was clicked until the timestamp, = 1 - (timestamp - the timestamp of the click) seconds/(30 * 60 * 1000); otherwise 0

"S_cy": screen central position y, always 0.5

"S_h": screen height, always 1

"MSG_y": if the message is visible on screen, MSG_y = the central of the message on screen (0 to 1), otherwise -1

"MSG_h": the message's visible height on screen, 0 to 1

"MSG_tclk": if the message was clicked at least once until the timestamp, = 1 - (timestamp - the timestamp of the click) seconds/(30 * 60 * 1000); otherwise 0

"height_1": the baseline 1, the message's visible height on screen, 0 to 1

"height_2": the baseline 2, 0.5 - the message's distance to the screen central. take value <=0.5

"height_3", the baseline 3 (mouse hover on the message),10000 if the mouse is hovering on the message; else 1/message's distance to the mouse);


#### Y (0/1):
"read", whether the participant is reading this message during the past second (0 or 1)
    
### session_feature_labeled.csv (1485 rows)    
Each row represents one participant's interaction with each message (story) in a reading session (open to close the post) and their read level (skip for 0/skim for 1/detail for 2)
#### Info columns:
userId,postId,index,n_word: as above

endtime: the timestamp when the participant finished reading this post

#### X:
MMF_y_inf, MMF_x_inf,: the participants' inf average mouse moving distance in the reading session

MSF_y_inf: the participants' inf average mouse scrolling distance in the reading session

MSF_clk: the percentage of messages clicked in the reading session

time_1,time_2,time_3: the total time the message is being read by the participant predicted by baseline 1/2/3

MSG_tmy: the message's total seconds under mouse during the session

MSG_sh: the message's average share on the screen during the reading session >0 when visible

MSG_sy: the message's average y position during the reading session >0 when visible

MSG_clk: whether the message is clicked during the session

MSG_svt: the number of seconds the message is visible on screen

#### Y:
read_level: the message's read level: (skip for 0/skim for 1/detail for 2)


## Model:
run EyeTrackingModel.py
