import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
import os


def resized_image(image):
    # Увеличиваем изображение
        scale_percent = 600
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized

def get_gray_filter_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray
# Размытие для уменьшения шумов
def blurred_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(f'blurred.png', blurred)
    return blurred

# Применение фильтра резкости
def sharpened_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(f'sharpened.png', sharpened)
    return sharpened

def get_skew_angle_via_hough(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    if len(angles) == 0:
        return 0
    return np.median(angles)

def rotate_image(image, angle):
    # Получаем размеры изображения
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Получаем матрицу поворота
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # Вычисляем новый размер изображения
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Центрируем изображение
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    # Поворачиваем изображение без обрезки
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

def find_counturs(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate = image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2.5 < aspect_ratio < 4.0 and w > 100 and h > 30:
            plate = image[y:y + h, x:x + w]
            return plate      
    return image
    


    
def clean_text(texts_numbers):
    """
    Очистка и исправление распознанного текста для российских номеров.
    """
    if len(texts_numbers) == 0:
        return None
    
    filter_texts = []
    for text in texts_numbers:
        if len(text) >= 6 and len(text) <= 9: 
            new_text = ""
            for i in range(len(text)):
                if i in [0, 4, 5]:
                    new_text += text[i].replace('8', 'В').replace('0', 'О').replace('7', 'У').replace('4', 'А').replace('6', 'В')
                else:
                    new_text += text[i].replace('В', '8').replace('О', '0').replace('У', '7').replace('Л', '1').replace('Б', '6').replace('А', '4').replace('Н', '4') 
            filter_texts.append(new_text)
            print(text)

    if len(filter_texts) == 0:
        return None
    
    pattern_texts = []
    for text in filter_texts:
        pattern1 = r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$'  #
        pattern2 = r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}$'
        if re.match(pattern1, text):
            pattern_texts.append(text)
        elif re.match(pattern2, text):
            pattern_texts.append(text)

    if len(pattern_texts) != 0:
        print(pattern_texts)
        return pattern_texts
    else:
        print("Олаала")
        print(filter_texts)
        return filter_texts


"""
Это основная функция, использовать напрямую если пользователь скинул изображение (не видео)
"""
def license_plate_detection_and_recognition(image, image_model):
    reader = easyocr.Reader(['ru'], gpu=False)
    cars_numbers = {} # Выборка номеров машин (в основном для виде исплользуеться)
    # Проходимся по изображениям
    for i, result in enumerate(image_model):
        car_numbers = []   # Выборка номеров машин
        boxes = result.boxes.xyxy.cpu().numpy()

        # Проходимся по найденным фрагментам
        for j, box in enumerate(boxes):

            # Способ 1
            # пробуем сначала увеличить размер фрагмента и его считать
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            resized_img = resized_image(cropped)

            # Считываем увеличенный фрагмент
            results = reader.readtext(resized_img, paragraph=True, allowlist='АВЕКМНОРСТУХ0123456789')
            if results:
                text_number = results[0][1].replace(" ", "").upper()
                car_numbers.append(text_number)
            cv2.imwrite(f"resized{j}.png", resized_img)

            # Способ 2
            # теперь переводим в оттенки серого и фильтруем
            gray_img = get_gray_filter_image(resized_img)
            blurred_img = blurred_image(gray_img)
            sharpened_img = sharpened_image(blurred_img)

            # Бинаризируем изображение
            _, binary = cv2.threshold(sharpened_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)

            # Находим угол наклона текста и поворачиваем фрагмент без обрезаний
            angle = get_skew_angle_via_hough(binary)
            deskewed = rotate_image(binary, angle)
            cv2.imwrite(f"deskewed1{j}.png", deskewed)

            plate_img = find_counturs(deskewed)
            cv2.imwrite(f"plate1{j}.png", plate_img)

            # считываем с нормальзованного фрагмента
            results = reader.readtext(plate_img, paragraph=True, allowlist='АВЕКМНОРСТУХ0123456789')
            if results:
                text_number = results[0][1].replace(" ", "").upper()
                car_numbers.append(text_number)

            print(car_numbers)
            # Способ 3
            # нормализация для светлого изображения
            binary_light = cv2.adaptiveThreshold(sharpened_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            deskewed = rotate_image(binary_light, angle)
            cv2.imwrite(f"deskewed2{j}.png", deskewed)

            plate_img = find_counturs(deskewed)
            cv2.imwrite(f"plate2{j}.png", plate_img)
            results = reader.readtext(plate_img, paragraph=True, allowlist='АВЕКМНОРСТУХ0123456789')
            if results:
                text_number = results[0][1].replace(" ", "").upper()
                car_numbers.append(text_number)

            # Способ 4
            # нормализация для тёмного изображения
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)

            _, binary_dark = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_dark = cv2.bitwise_not(binary_dark)

            deskewed = rotate_image(binary_dark, angle)
            cv2.imwrite(f"deskewed3{j}.png", deskewed)

            plate_img = find_counturs(deskewed)
            cv2.imwrite(f"plate3{j}.png", plate_img)

            results = reader.readtext(plate_img, paragraph=True, allowlist='АВЕКМНОРСТУХ0123456789')
            if results:
                text_number = results[0][1].replace(" ", "").upper()
                car_numbers.append(text_number)

        # Фильтруем и выбераем наилучшие варианты номеров
        car_numbers = clean_text(car_numbers)
        if car_numbers:
            cars_numbers[f"{i+1}"] = car_numbers


    return cars_numbers

"""
Это для видео, он потом проходится по каждому изображению и обрабатывает их в функции license_plate_detection_and_recognition
"""
def video_detection_and_recognition(model_yolo, video):
    output_dir = 'frames'

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = 0
    saved_frame_count = 0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # конец видео

        # Сохраняем только каждый N-й кадр (например, каждый 30-й)
        if frame_number % int(fps) == 0:
            frame_name = f'frame_{saved_frame_count:02d}.jpg'
            frame_filename = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

            frames.append(frame_name)

        frame_number += 1

    cap.release()

    cars_numbers = {}
    for i in range(len(frames)):
        path_frame = f"{output_dir}/{frames[i]}"
        img = cv2.imread(path_frame)
        img_mod = model_yolo(img)
        numbers = license_plate_detection_and_recognition(img, img_mod)
        if len(numbers) > 0:
            cars_numbers[i] = numbers[i]

    return cars_numbers


if __name__ == "__main__":
    model = YOLO("runs/detect/train3/weights/best.pt")    # Путь до нашей модели Yolo

    image_path = 'test1.jpg'
    img = cv2.imread(image_path)
    img_mod = model(img)

    numbers = license_plate_detection_and_recognition(img, img_mod)
    if len(numbers) > 0:
        for key, value in numbers.items():
            print(f"=== Машина №{key} ===")
            for number in value:
                print(number)
    else:
        print("модель ии не смогла распознать номер, скидывайте в хорошом качестве снимки!")
