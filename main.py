import logging
import concurrent.futures
import arabic_reshaper
import cv2
from bidi.algorithm import get_display
from ultralytics import YOLO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os

yolo_model = YOLO("/Users/mernaziad/Desktop/final_train/runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

font_face = cv2.FONT_HERSHEY_SIMPLEX
font_color = (255, 255, 255)
thickness = 2
font_path = "./arabic_font.ttf"
font_size = 32  # Adjust the font size as needed
# Set the font properties
font_scale = 1
font_thickness = 2

# Set up logging
logging.basicConfig(level=logging.INFO)

translation_dict = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    'ALIF': 'أ',
    'AYN': 'ع',
    'Atman lak 7aya sa3eeda': 'أتمنى لك حياة سعيدة',
    'BAA': 'ب',
    'DAD': 'ض',
    'DELL': 'د',
    'DHAA': 'ظ',
    'DHELL': 'ذ',
    'FAA': 'ف',
    'GHAYN': 'غ',
    'HA': 'هـ',
    'HAA': 'ح',
    'JEEM': 'ج',
    'KAAF': 'ك',
    'KHAA': 'خ',
    'LAAM': 'ل',
    'MEEM': 'م',
    'QAAF': 'ق',
    'RAA': 'ر',
    'SAD': 'ص',
    'SEEN': 'س',
    'SHEEN': 'ش',
    'TA': 'ت',
    'TAA': 'ط',
    'THA': 'ث',
    'WAW': 'و',
    'YA': 'ي',
    'ZAY': 'ز',
    'bad': 'سيء',
    'del': 'حذف',
    'eqtibas': 'اقتباس',
    'good': 'جيد',
    'law sama7t': 'لو سمحت',
    'merhaba': 'مرحبا',
    'nothing': 'لا شيء',
    'o7ebok': 'أحبك',
    'oraqebak': 'أراقبك',
    'space': 'مسافة',
    'you': 'أنت',
}


def translate_to_arabic(text):
    if text in translation_dict:
        return translation_dict[text]
    else:
        return text  # إذا لم تكن الكلمة موجودة في القاموس، استخدم النص الأصلي


def convert_text_to_image(text, background_color=(0, 0, 0, 0)):
    # Load the specified font
    font = ImageFont.truetype(font_path, font_size)

    # Create a new image with a transparent background
    image = Image.new('RGBA', (1, 1), background_color)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Calculate the text size
    text_width, text_height = draw.textsize(text, font=font)

    # Add some padding to the text size
    padding = 10
    image_width = text_width + padding
    image_height = text_height + padding

    # Create a new image with the adjusted dimensions and background color
    image = Image.new('RGBA', (image_width, image_height), background_color)

    # Create a new draw object with the adjusted image
    draw = ImageDraw.Draw(image)

    # Calculate the position to place the text at the bottom of the image
    text_x = (image.width - text_width) // 2
    text_y = image.height - text_height - padding

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=font_color)

    # Convert the PIL image to RGB mode
    image = image.convert("RGB")

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    return image_array


def get_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text


def overlay_text_on_frame(frame, text_image):
    padding = 10
    y = frame.shape[0] - text_image.shape[0] - padding
    x = (frame.shape[1] - text_image.shape[1]) // 2
    frame[y:y + text_image.shape[0], x:x + text_image.shape[1]] = text_image

    return frame


def detect_objects(frame):
    results = yolo_model.predict(frame)
    result = results[0]
    frame_with_text = frame
    texts_to_speak = []
    for box in results[0].boxes:
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = result.names[box.cls[0].item()]
        translated_text = translate_to_arabic(class_id)
        expected_text = get_arabic_text(translated_text)

        conf = round(box.conf[0].item(), 2)
        logging.info("Object type:{} {}".format(expected_text, class_id))
        logging.info("Coordinates: {}".format(cords))
        logging.info("Probability: {}".format(conf))

        if conf < 0.5:
            return

        text_image = convert_text_to_image(expected_text)
        frame_with_text = overlay_text_on_frame(frame, text_image)

        # Convert the Arabic text to speech
        tts = gTTS(text=translated_text, lang='ar')
        audio_file = "/Users/mernaziad/Desktop/text_to_speech.mp3"
        tts.save(audio_file)

        # Play the Arabic text as speech
        os.system('mpg123 ' + audio_file)
        os.remove(audio_file)

        texts_to_speak.append(translated_text)

    # Display the modified frame
    cv2.imshow("Object Detection", frame_with_text)


def process_frames():
    ret, frame = cap.read()
    if frame.shape[:2] != (800, 600):
        frame = cv2.resize(frame, (800, 600))
    # Perform object detection and modification on the frame
    detect_objects(frame)
    # Break the loop if 'q' is pressed


while True:
    process_frames()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
# Submit the process_frames function to the executor
executor.submit(process_frames)
# Wait for the 'q' key to be pressed
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass

# Shutdown the executor
executor.shutdown()
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

