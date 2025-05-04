import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from deepface import DeepFace
import pygame

pygame.init()

known_image = face_recognition.load_image_file("owner.png")
known_encoding = face_recognition.face_encodings(known_image)[0]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

success, frame = cap.read()
if not success:
    print("Не удалось захватить кадр")
    cap.release()
    pygame.quit()
    exit()

height, width, _ = frame.shape
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Веб-камера")

running = True
while running:
    success, frame = cap.read()
    if not success:
        print("Ошибка захвата кадра")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_location
        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]

        label = "неизвестный"

        hand_data = hands.process(rgb_frame)  
        fingers_up = 0

        if hand_data.multi_hand_landmarks:
            hand_landmarks = hand_data.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            tips = [4, 8, 12, 16, 20]
            fingers_up = 0
            for tip in tips[1:]:
                if lm[tip].y < lm[tip - 2].y:
                    fingers_up += 1
            if lm[4].x > lm[3].x:
                fingers_up += 1

            mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if match:
            if fingers_up == 1:
                label = "сашка"
            elif fingers_up == 2:
                label = "трофимов"
            elif fingers_up == 3:
                face_crop = frame[top:bottom, left:right]  # DeepFace ожидает BGR
                try:
                    emotion_result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    label = emotion_result[0]["dominant_emotion"]
                except Exception as e:
                    label = "эмоция не определена"
        else:
            label = "неизвестный"

        cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Зеленый в RGB
        cv2.putText(rgb_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # Белый
        print(f"[LOG] Обнаружено лицо: {label}, Пальцев: {fingers_up}")

    surface = pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")
    screen.blit(surface, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    pygame.display.flip()

cap.release()
pygame.quit()
