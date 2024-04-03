import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import bcrypt
import mysql.connector
import os

# Database initialization
conn = mysql.connector.connect(
    host="localhost",
    user="Jash",
    password="jash123",
    database="login_signup"
)
cursor = conn.cursor()

model  = load_model(r"C:\Users\jashp\Downloads\emotion-based-music-main\emotion-based-music-main\model.h5")
label = np.load(r"C:\Users\jashp\Downloads\emotion-based-music-main\emotion-based-music-main\labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Function to signup
def signup(username, password):
    # Hash the password before storing it in the database
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cursor.execute('''INSERT INTO user (username, password_hash) VALUES (%s, %s)''', (username, password_hash))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False  # Username already exists

# Function to login
def login(username, password):
    cursor.execute('''SELECT password_hash FROM user WHERE username = %s''', (username,))
    result = cursor.fetchone()
    if result:
        stored_password_hash = result[0]
        # Check if the provided password matches the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return True
    return False

# Function for signup page
def signup_page():
    st.markdown("<h1 style='color: white;'>Signup Page</h1>", unsafe_allow_html=True)

    # Signup form
    st.markdown("<p style='color: white;'>Username (Signup)</p>", unsafe_allow_html=True)
    signup_username = st.text_input("", key="signup_username")
    st.markdown("<p style='color: white;'>Password (Signup)</p>", unsafe_allow_html=True)
    signup_password = st.text_input("", type="password", key="signup_password")

    if st.button("Signup"):
        if signup_username and signup_password:
            if signup(signup_username, signup_password):
                st.success("Signup successful! Please login.")
                return "login"
            else:
                st.error("Username already exists.")
    return None

# Function for login page
def login_page():
    st.markdown("<h1 style='color: white;'>Login Page</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>Username (Login)</p>", unsafe_allow_html=True)
    login_username = st.text_input("", key="login_username")

    st.markdown("<p style='color: white;'>Password (Login)</p>", unsafe_allow_html=True)
    login_password = st.text_input("", type="password", key="login_password")

    if st.button("Login"):
        if login_username and login_password:
            if login(login_username, login_password):
                st.success("Login successful! Redirecting to main app...")
                return "main_app"
            else:
                st.error("Invalid username or password.")
    return None

# Function for main app (emotion recommender)
def main_app():
    st.markdown("<h1 style='color: white;'>MoodVibes</h1>", unsafe_allow_html=True)

    class EmotionProcessor:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")

            ##############################
            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]

                print(pred)
                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                np.save("emotion.npy", np.array([pred]))


            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


            ##############################

            return av.VideoFrame.from_ndarray(frm, format="bgr24")

    st.markdown("<h2 style='color: white;'>Enter Language</h2>", unsafe_allow_html=True)
    lang = st.text_input("", key="language")

    # Custom HTML to set the title color to white
    st.markdown("<h2 style='color: white;'>Enter Singer</h2>", unsafe_allow_html=True)
    singer = st.text_input("", key="singer")

    if lang and singer:
        webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

    if st.button("Recommend me songs"):
        emotion_data = np.load("emotion.npy")
        if emotion_data is not None and len(emotion_data) > 0:
            emotion = emotion_data[0]
            if emotion:
                url = f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
                webbrowser.open(url)
                np.save("emotion.npy", np.array([""]))
            else:
                st.warning("Please let me capture your emotion first")
        else:
            st.warning("Please let me capture your emotion first")





def set_background_image():
    """
    Function to set background image using custom HTML/CSS
    """
    # URL of the background image
    image_url = "https://www.gifcen.com/wp-content/uploads/2022/06/lofi-gif-7.gif"
    
    # Custom CSS for setting background image
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
    }}
    </style>
    """
    # Inject the custom CSS
    st.markdown(background_style, unsafe_allow_html=True)

def main():
    # Set background image
    set_background_image()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_app()
    else:
        # Show signup page first
        page = signup_page()
        if page == "login":
            st.experimental_rerun()
        page = login_page()

        if page == "main_app":
            st.session_state.logged_in = True
            main_app()

if __name__ == "__main__":
    main()
