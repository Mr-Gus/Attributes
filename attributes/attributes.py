from deepface import DeepFace
from cv2 import *
import cv2
import time


profile = []

def take_pic(img_name):
    try:
        cam = cv2.VideoCapture(-1)
        retval, frame = cam.read()
        if retval != True:
            raise ValueError("Can't read frame")

        cv2.imwrite(img_name, frame)
        cv2.imshow("Image", frame)

    except Exception:
        pass


#analyze face expressions
def facial_emotions():
    try:

        analysis = DeepFace.analyze(img_path = img_name, actions = ["age", "gender", "emotion", "race"])
        emotions = analysis['emotion']
        age = analysis['age']
        gender = analysis['gender']
        race = analysis['race']
        
        total = 0
        race_total = 0
        top3_moods = []
        probable_race = []

        race = dict(sorted(race.items(),key=lambda x:x[1], reverse=True))
        for race, percent in race.items():
            probable_race.append({race:'{:.2f}%'.format(percent)})
            race_total += 1
            if race_total == 2: 
                race_total = 0
                break

        emotions = dict(sorted(emotions.items(), key=lambda x:x[1], reverse=True))
        for emotion, percent in emotions.items():
            top3_moods.append({emotion:'{:.2f}%'.format(percent)})
            total += 1
            if total == 3: 
                total = 0
                break
        

        if gender:
            profile.append({'image': img_name, 'gender': gender, 'race': probable_race, 'age': age, 'mood': top3_moods})
        
        print(profile[-1])
        return profile

    except Exception:    
        pass


while True:
    
    img_name = f'database/img{len(profile)}.jpg'
    
    take_pic(img_name)
    facial_emotions()
    time.sleep(5)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

"""
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
DeepFace.stream(db_path='database', model_name = models[0])
"""