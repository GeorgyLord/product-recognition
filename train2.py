from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv12 Training')
    parser.add_argument('--epochs', type=int, default=2, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--device', default=0, help='Устройство (cpu или 0,1,2,3 для GPU)')
    parser.add_argument('--model', default='yolo12n.pt', help='Модель для обучения')
    
    args = parser.parse_args()
    
    # Загрузка модели
    model = YOLO(args.model)
    
    # Обучение
    results = model.train(
        # Можно выбрать:
        # dataset/data или dataset_2/data
        data='dataset_2/data.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=15,
        save=True,
        exist_ok=True
    )
    
    # Валидация
    metrics = model.val()
    print(f"Результаты валидации: {metrics}")
    
    # Экспорт модели
    # model.export(format='onnx')  # Можно экспортировать в ONNX, TensorRT и др.

if __name__ == "__main__":
    main()