Тествое задание на должность Инженер-программист CV в компанию Техтранс

Задача:
- выбрать из tensorflow object detection zoo модель пригодную для определения расположения человека в кадре (можно использовать keras);
- выбрать произвольный ролик на несколько секунд где присутствуют люди;
- с использованием установленного из исходников opencv-4.4.0, python3.7 и tensorflow-2.1 пропустить выбранный ролик через выбранную модель и записать результат в файл.

Результатом работы должен быть:
- github репозиторий, с историей commit-ов в котором будут присутствовать входящий видеофайл;
- выходящий видеофайл; 
- файл requirements.txt со списком необходимых для работы программы пакетов;
- сохраненная модель или скрипт скачивающий и подготавливающий модель к использованию; 
- скрипт который может запустить обработку указанного как аргумент входящего видеофайла.

Описание выполненной работы

Установленные пакеты и среды:
- ubuntu 20.04
- pycharm 2022.1.1
- python 3.8.10
- matplotlib 3.5.2
- numpy 1.22.4
- opencv_python 4.5.5.64
- Pillow 9.1.1
- tensorflow_cpu 2.9.1

Описание структуры проекта:
- data - раздел для хранения тестовых файлов;
- pretrain_models - раздел для хранения предтренированных моделей из репозитория model zoo object detection;
- create_model.ipynb - jupyter файл с описанием и оценкой применения нескольких моделей
- main.py - исполняемый файл, который запускает скачивание модели, осуществляет поиск на видео людей и записывает результаты в файл video_predict.mp4 в корневом каталоге проекта;
- requirements.txt - устанавливаемые зависимости.

Для запуска скрипта необходимо клонировать репозиторий проекта, установить зависимости и ввести комманду: 
main.py data/video2.mp4
где data/video2.mp4 - адрес расположения видео, на котором необходимо определить объекты
