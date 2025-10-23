from ultralytics import YOLO

def test_trained_model():
    # Загрузка обученной модели
    model = YOLO('runs/detect/train/weights/best.pt')
    
    # Предсказание на тестовом изображении
    results = model.predict('folder_for_testing/8.jpg', save=True, conf=0.5)
    
    # Отображение результатов
    for r in results:
        # print(r)
        im_array = r.plot()
        # Сохранение изображения с детекциями
        from PIL import Image
        im = Image.fromarray(im_array[..., ::-1])
        im.save('result.jpg')

if __name__ == "__main__":
    test_trained_model()