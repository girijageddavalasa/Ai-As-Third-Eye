import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Girija Pro\TesseractOCR\tesseract.exe' # Replace with your path

def capture_and_extract():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        if text:
            print("Text detected:", text)
            cap.release()
            cv2.destroyAllWindows()
            return text
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    return None

print(capture_and_extract())
