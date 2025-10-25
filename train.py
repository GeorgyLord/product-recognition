from ultralytics import YOLO
import os

def train_yolov12():
    # Путь к конфигурационному файлу данных
    data_yaml = "dataset_2/data.yaml"
    
    # Проверяем существование файла data.yaml
    if not os.path.exists(data_yaml):
        print(f"Ошибка: Файл {data_yaml} не найден!")
        return
    
    # Загружаем модель YOLOv12
    # Если YOLOv12 недоступен, используйте последнюю версию YOLO
    try:
        model = YOLO('yolo12n.pt')  # или 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt'
    except:
        print("YOLOv12 не найден, используем YOLOv11")
        model = YOLO('yolo11n.pt')
    
    # Параметры обучения
    training_params = {
        'data': data_yaml,
        'epochs': 10,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',  # или '0' для GPU, 'cpu' для CPU
        'resume': False,
        'workers': 4,
        'patience': 10,
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }
    
    # Запуск обучения
    results = model.train(**training_params)
    
    # Сохранение результатов
    print("Обучение завершено!")
    return results

if __name__ == "__main__":
    train_yolov12()