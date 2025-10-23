from ultralytics import YOLO

def predict_on_video_simple(model_path, video_path):
    # Загрузка модели
    model = YOLO(model_path)
    
    # Предсказание на видео
    results = model.predict(
        source=video_path,
        conf=0.5,        # порог уверенности
        save=True,       # сохранить результат
        show=True,       # показать в реальном времени
        project='results',  # папка для результатов
        name='video_detection'
    )
    
    return results

# Использование
predict_on_video_simple('runs/detect/train/weights/best.pt', 'folder_for_testing/test_video.mp4')