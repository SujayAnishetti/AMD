import cv2
image_path = "11.PNG"
img = cv2.imread(image_path)

def print_pixel_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img[y, x]
        print(f"Pixel at ({x}, {y}) - BGR: ({b}, {g}, {r})")

cv2.namedWindow("image")
cv2.setMouseCallback("image", print_pixel_values)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
