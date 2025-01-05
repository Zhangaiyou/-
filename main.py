import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# 数据准备
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label in range(10):
        label_folder = os.path.join(folder_path, str(label))
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  # 确保图像尺寸为28x28
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


# 加载训练和测试数据
train_images, train_labels = load_images_from_folder('mnist_train')
test_images, test_labels = load_images_from_folder('mnist_test')

# 数据重塑
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# 数据格式转换
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))


accuracy_history = AccuracyHistory()

# 构建CNN模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=50,
                    batch_size=64,
                    callbacks=[accuracy_history])

# 绘制准确率变化图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracy_history.accuracy) + 1), accuracy_history.accuracy, label='Training Accuracy')
plt.plot(range(1, len(accuracy_history.val_accuracy) + 1), accuracy_history.val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([min(min(accuracy_history.accuracy), min(accuracy_history.val_accuracy)) * 0.95, 1])  # 设置y轴范围，避免顶部被截断
plt.show()

# 预测测试集标签
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# 计算混淆矩阵
conf_mat = confusion_matrix(test_labels.argmax(axis=1), predicted_classes)

# 打印混淆矩阵
print("Confusion Matrix:\n", conf_mat)

# 计算并打印分类报告
class_report = classification_report(test_labels.argmax(axis=1), predicted_classes)
print("\nClassification Report:\n", class_report)

# 评估模型
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存模型
model.save('mnist_cnn_model.keras')
