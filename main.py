import argparse
import tensorflow as tf
import cv2
import pathlib
import numpy as np

# Создаем объект парсера и считываем аргументы при вызове скрипта predict.py
parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
a = parser.parse_args().__dict__

TRESHOLD = 0.4
CLASS_ID = 1
VIDEO = a['video_path']
VIDEO_OUTPUT = './video_predict.mp4'


def load_model(model_name):
    """ Функция принимает на вход название модели, скачивает из
    репозитория model zoo в рабочую директорию 'pretrain_models',
    поcле чего возвращает ее в формает tensorflow
    """
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                    origin=base_url + model_file, untar=True,
                    cache_dir='./', cache_subdir='pretrain_models')
    model_dir = pathlib.Path(model_dir+'/saved_model')
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

# загрузим модель centernet_resnet50_v1

detection_centernet_resnet50_v1_fpn_512x512 = load_model('centernet_resnet50_v1_fpn_512x512_coco17_tpu-8')


def predict_frame(frame, frame_width, frame_height):
    """Функция принимает на вход фрейм, создает предсказания
    и выбирает только те, где есть люди, после чего возвращает исходное
    изображение с  предсказанными bbox.
    frame - кадр;
    frame_width, frame_height - ширина и высота кадра.
    """
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = detection_centernet_resnet50_v1_fpn_512x512(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                     for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    for i,n in enumerate(output_dict['detection_classes']):
        if n == CLASS_ID and output_dict['detection_scores'][i] > TRESHOLD:
            b = output_dict['detection_boxes'][i]
            frame = cv2.rectangle(frame,(int(b[1]*frame_width), int(b[0]*frame_height)),(int(b[3]*frame_width),int(b[2]*frame_height)),(0,255,0),3)
    return frame

# Функция записывает предсказания  файл
def write_predict(video_path):
    """Функция принимает на вход путь к видеофайлу,
    определяет расположение людей и записывает результат в видеофайл mp4
    video_path: путь к выходному видео.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 24, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = predict_frame(frame, frame_width, frame_height)
            output.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            break

            cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    write_predict(VIDEO)


