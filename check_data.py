import yaml
import os

def check_dataset_structure():
    # Проверяем data.yaml
    # Можно выбрать:
    # dataset/data или dataset_2/data
    with open('dataset_2/data.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    print("Структура датасета:")
    print(f"Количество классов: {data['nc']}")
    print(f"Имена классов: {data['names']}")
    
    # Проверяем существование папок
    folders = ['dataset/train', 'dataset/valid', 'dataset/test']
    for folder in folders:
        if os.path.exists(folder):
            images_dir = os.path.join(folder, 'images')
            labels_dir = os.path.join(folder, 'labels')
            
            if os.path.exists(images_dir):
                images_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"{folder}/images: {images_count} изображений")
            
            if os.path.exists(labels_dir):
                labels_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                print(f"{folder}/labels: {labels_count} файлов разметки")
        else:
            print(f"Папка {folder} не найдена!")

if __name__ == "__main__":
    check_dataset_structure()