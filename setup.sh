#!/bin/bash

# setup.sh - Автоматическая настройка окружения для YOLO проекта

echo "🚀 Начинаем настройку проекта..."

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен. Установите Python3 сначала."
    exit 1
fi

# Создаем виртуальное окружение
echo "📦 Создаем виртуальное окружение..."
python -m venv .venv

# Активируем виртуальное окружение
echo "🔧 Активируем виртуальное окружение..."
source .venv/bin/activate

# Обновляем pip
echo "🔄 Обновляем pip..."
pip install --upgrade pip

# Устанавливаем основные зависимости для YOLO
echo "📚 Устанавливаем основные библиотеки..."
pip install ultralytics
pip install opencv-python
pip install pillow
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install scipy

# Устанавливаем дополнительные утилиты
# echo "📦 Устанавливаем дополнительные утилиты..."
# pip install jupyter
# pip install ipython
# pip install tqdm
# pip install albumentations
# pip install wandb

# Устанавливаем зависимости для экспорта моделей
echo "🔧 Устанавливаем зависимости для экспорта..."
pip install onnx
# pip install onnxruntime
# pip install tensorboard

# Создаем необходимые папки
# echo "📁 Создаем структуру папок..."
# mkdir -p data/raw
# mkdir -p data/processed
# mkdir -p models
# mkdir -p utils
# mkdir -p configs
# mkdir -p runs/detect

# Создаем базовые конфигурационные файлы
echo "⚙️ Создаем конфигурационные файлы..."

# requirements.txt для будущего использования
cat > requirements.txt << EOL
ultralytics>=8.0.0
opencv-python>=4.5.0
Pillow>=9.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
torch>=1.10.0
torchvision>=0.11.0
EOL

# Создаем .gitignore
cat > .gitignore << EOL
# Virtual environment
.venv/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg-info/

# Data and models
data/raw/
data/processed/
models/
*.pt
*.pth
*.onnx

# Training results
runs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
# .ipynb_checkpoints/
# EOL

# echo "✅ Настройка завершена!"
# echo ""
# echo "📝 Для активации окружения выполните:"
# echo "   source .venv/bin/activate"
# echo ""
# echo "🐍 Для проверки установки выполните:"
# echo "   python -c \"from ultralytics import YOLO; print('YOLO установлен успешно!')\""
# echo ""
# echo "🎯 Для запуска обучения:"
# echo "   python train.py"