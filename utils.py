import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_files


def load_dataset(path):
    data = load_files(path, load_content=False, random_state=42)
    df = pd.merge(pd.DataFrame(data.target, columns=['target']), pd.DataFrame(data.target_names, columns=['label']),
                  how='left', left_on='target', right_index=True)
    df = pd.concat([df, pd.DataFrame(data.filenames, columns=['filename'])], axis=1, sort=False)
    return df


def imshow(img):
    plt.imshow(img)
    plt.show()


def show_first_img(df):
    fig = plt.figure(figsize=[20, 20])
    for idx, filename in enumerate(df['filename'].values[:10]):
        img = mpimg.imread(filename)
        ax = fig.add_subplot(5, 5, idx + 1)
        plt.imshow(img)
        ax.set_title(df['label'].values[idx])

    plt.show()


def detect_human_face(file, show=True):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # load color (BGR) image
    img = cv2.imread(file)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # print number of faces detected in the image

    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show:
        # display the image, along with bounding box
        ax = plt.axes()
        ax.set_title(f'Number of faces detected: {len(faces)}')
        plt.imshow(cv_rgb)
        plt.show()
    else:
        return cv_rgb, len(faces)


def detect_human_faces(file_list):
    fig = plt.figure(figsize=[20, 20])
    for idx, file in enumerate(file_list):
        cv_rgb, num_faces = detect_human_face(file, show=False)
        ax = fig.add_subplot(5, 5, idx + 1)
        plt.imshow(cv_rgb)
        ax.set_title(f'Num faces detected: {num_faces}')

    plt.show()

    
def face_detector(img_path):
    '''returns "True" if face is detected in image stored at img_path'''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0