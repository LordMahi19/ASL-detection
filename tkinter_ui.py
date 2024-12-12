import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
from gtts import gTTS
from io import BytesIO
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pygame
from PIL import Image, ImageTk


def process_frame(frame_queue, result_queue, model):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                result_queue.put((frame, predicted_character, (x1, y1, x2, y2)))
            except ValueError:
                result_queue.put((frame, None, (x1, y1, x2, y2)))
        else:
            result_queue.put((frame, None, None))


def main():
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)
    frame_queue = Queue()
    result_queue = Queue()

    process = Process(target=process_frame, args=(frame_queue, result_queue, model))
    process.start()

    def on_closing():
        frame_queue.put(None)  
        cap.release() 
        root.destroy()

    def update_ui():
        nonlocal current_char, timer, sentence, spoken_sentences
        ret, frame = cap.read()
        if not ret:
            root.after(10, update_ui)
            return

        frame_queue.put(frame)
        processed_frame, detected_char, bbox = result_queue.get()

        if detected_char == current_char:
            timer += 1
        else:
            timer = 0
            current_char = detected_char

        if timer >= 30 and current_char:
            if current_char == 'space':
                sentence += ' '
                audio_buffer = BytesIO()
                speech = gTTS(text='space', lang='en', slow=False)
                speech.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                pygame.mixer.music.load(audio_buffer, 'mp3')
                pygame.mixer.music.play()
            elif current_char == 'del':
                if sentence:
                    spoken_sentences.append(sentence)
                    sentences_box.delete(1.0, tk.END)
                    sentences_box.insert(tk.END, "\n".join(spoken_sentences))

                    audio_buffer = BytesIO()
                    speech = gTTS(text=sentence, lang='en', slow=False)
                    speech.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_buffer, 'mp3')
                    pygame.mixer.music.play()

                    sentence = ''

            else:
                sentence += current_char
                audio_buffer = BytesIO()
                char_speech = gTTS(text=current_char, lang='en', slow=False)
                char_speech.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                pygame.mixer.music.load(audio_buffer, 'mp3')
                pygame.mixer.music.play()

            timer = 0

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            if detected_char:
                cv2.putText(processed_frame, 'ENTER' if detected_char == 'del' else detected_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, sentence)

        progress['value'] = (timer / 30) * 100

        frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = int(screen_width * 0.6) 
        new_height = int(new_width / aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_label.config(image=img)
        video_label.image = img

        root.after(10, update_ui)

    root = tk.Tk()
    root.title("Sign Language Detector")
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

    root.protocol("WM_DELETE_WINDOW", on_closing)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    style = ttk.Style()
    style.configure("TFrame", background="lightblue")
    style.configure("TLabel", background="lightblue", font=("Arial", 18))
    style.configure("TProgressbar", thickness=20)
    style.configure("delete.TButton", background="red", foreground="red")
    style.configure("clear.TButton", background="green", foreground="green")
    style.configure("view.TButton", background="blue", foreground="blue")

    # Main Layout
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    button_frame = ttk.Frame(left_frame, width=300)
    button_frame.pack(fill=tk.BOTH, pady=5)

    def delete_last_character():
        nonlocal sentence
        if sentence:
            sentence = sentence[:-1]

    def clear_text():
        nonlocal sentence
        sentence = ""

    def view_signs():
        """Open a new window displaying an image of sign language gestures."""
        new_window = tk.Toplevel(root)
        new_window.title("Sign Language Reference")
        new_window.geometry("1280x720")

        # Load and display an example image of sign language
        img = Image.open("signs.png")  # Replace with your image path
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        img_label = tk.Label(new_window, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack(fill=tk.BOTH, expand=True)

    # Create buttons
    btn_delete = ttk.Button(button_frame, text="DELETE", style="delete.TButton", command=delete_last_character)
    btn_delete.pack(side=tk.LEFT, padx=20, pady=20, expand=True)

    btn_clear = ttk.Button(button_frame, text="CLEAR", style="clear.TButton",command=clear_text)
    btn_clear.pack(side=tk.LEFT, padx=20, pady=20, expand=True)

    btn_view_signs = ttk.Button(button_frame, text="VIEW SIGNS", style="view.TButton", command=view_signs)
    btn_view_signs.pack(side=tk.LEFT, padx=20, pady=20, expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Real-time text display
    text_frame = ttk.Frame(left_frame, width=300) 
    text_frame.pack(fill=tk.BOTH, pady=5)

    ttk.Label(text_frame, text="Real-Time Text").pack(anchor=tk.W)
    text_output = tk.Text(text_frame, height=5, width=40, font=("Arial", 18)) 
    text_output.pack(pady=5, fill=tk.BOTH, expand=False) 

    progress = ttk.Progressbar(text_frame, mode='determinate')
    progress.pack(pady=5, fill=tk.X) 

    # Camera feed
    camera_frame = ttk.Frame(right_frame)
    camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    ttk.Label(camera_frame, text="Camera Feed").pack(anchor=tk.W)
    video_label = tk.Label(camera_frame)
    video_label.pack(fill=tk.BOTH, expand=True)

    # Spoken sentences display
    sentences_frame = ttk.Frame(left_frame, width=300)  
    sentences_frame.pack(fill=tk.BOTH, pady=5)

    ttk.Label(sentences_frame, text="Spoken Sentences").pack(anchor=tk.W)
    sentences_box = tk.Text(sentences_frame, font=("Arial", 18), width=40) 
    sentences_box.pack(pady=5, fill=tk.BOTH, expand=False) 

    pygame.init()

    current_char = None
    timer = 0
    sentence = ''
    spoken_sentences = []

    root.after(10, update_ui)
    root.mainloop()

if __name__ == '__main__':
    main()
