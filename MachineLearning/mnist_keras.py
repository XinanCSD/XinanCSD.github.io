"""
keras 构建神经网络实现手写数据识别
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import argparse
import glob
import sys
import os
import cv2


class Setting:
    """配置类"""
    def __init__(self):
        self.data_dir = 'mnist_data/'
        self.train_dir = self.data_dir + 'train'
        self.test_dir = self.data_dir + 'test'
        self.img_width = 28
        self.img_height = 28
        if backend.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 3)
        self.checkpoint_path = self.data_dir + 'model_weight/weights-improvement-{epoch:02d}-{val_acc:.5f}.hdf5'

        self.restore_model = None
        self.is_restore = False
        self.imp_dir = self.data_dir + 'tmp'

        # 模型训练的配置
        self.lebel_nums = 10
        self.epoches = 20
        self.save_epoches = 1
        self.batch_size = 32
        self.initial_epoch = 0


class MNISTClassifier:
    def __init__(self, setting: Setting):
        self.setting = setting

    def data_generator(self, setting: Setting):
        """图像数据生成器，对生成的字体图像划分训练集和验证集，同时进行数据增强，提高模型泛化能力。"""
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )
        train_generator = datagen.flow_from_directory(
            setting.train_dir,
            target_size=(setting.img_width, setting.img_height),
            batch_size=setting.batch_size,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = datagen.flow_from_directory(
            setting.train_dir,
            target_size=(setting.img_width, setting.img_height),
            batch_size=setting.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        return train_generator, validation_generator

    def build_model(self):
        """构建网络模型"""
        setting = self.setting
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=setting.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(setting.lebel_nums))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def checkpoint_callbacks(self):
        """保存模型的回调函数"""
        setting = self.setting
        checkpoint = ModelCheckpoint(
            setting.checkpoint_path, monitor='val_acc',
            save_weights_only=True, verbose=1,
            save_best_only=True, mode='max', period=setting.save_epoches)
        callbacks_list = [checkpoint]
        return callbacks_list

    def train(self):
        """训练模型"""
        print(':::Tring model:::')
        setting = self.setting
        train_generator, validation_generator = self.data_generator(setting) # 可以只从少量数据进行训练
        train_samples = train_generator.samples
        steps_per_epoch = max(125, train_samples // setting.batch_size)

        callbacks_list = self.checkpoint_callbacks()

        model = self.build_model()

        if setting.is_restore:
            print("导入模型继续训练：{} ".format(setting.restore_model))
            model.load_weights(setting.restore_model)

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=setting.epoches,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch,
            shuffle=True,
            initial_epoch=setting.initial_epoch,
            callbacks=callbacks_list,
            verbose=1,
            use_multiprocessing=True
        )

    def evaluate(self):
        """评估模型"""
        print(':::Evaluating:::')
        setting = self.setting

        models_weights = glob.glob(setting.data_dir + 'model_weight/*.hdf5')
        if len(models_weights) == 0:
            print('没有保存的模型，重新训练！')
            self.train()
            models_weights = glob.glob(setting.data_dir + 'model_weight/*.hdf5')
        models_weights.sort()
        model = self.build_model()
        restore_model = models_weights[-1]
        model.load_weights(restore_model)
        test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_data_gen.flow_from_directory(
            setting.test_dir,
            target_size=(setting.img_width, setting.img_height),
            batch_size=setting.batch_size,
            class_mode='categorical',
        )
        ret = model.evaluate_generator(test_generator,
                                       steps=125)
        print("val_loss:{}, val_acc:{}.".format(ret[0], ret[1]))

    def predict(self):
        """应用模型进行预测"""
        print(':::Applying predicting:::')
        setting = self.setting
        model = self.build_model()

        models_weights = glob.glob(setting.data_dir + 'model_weight/*.hdf5')
        if len(models_weights) == 0:
            print('没有保存的模型，重新训练！')
            self.train()
            models_weights = glob.glob(setting.data_dir + 'model_weight/*.hdf5')
        models_weights.sort()

        restore_model = models_weights[-1]
        model.load_weights(restore_model)

        input_shape = [1, setting.img_width, setting.img_height, 3]
        for name in os.listdir(setting.imp_dir):
            if name == '.DS_Store':
                continue
            img_file = os.path.join(setting.imp_dir, name)
            print(img_file)
            img = cv2.imread(img_file)
            imarr = img.reshape(input_shape)
            imarr = imarr / 255
            ret_np = model.predict(imarr)
            parser_val = ret_np.argmax()
            print('预测结果为：{}'.format(parser_val))


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='must be train|evaluate|predict')
    parser.add_argument('--is_restore', required=False, default=False)
    parser.add_argument('--restore_model', required=False)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = args_parser()
    setting = Setting()
    setting.is_restore = args['is_restore']
    setting.restore_model = args['restore_model']

    mnist = MNISTClassifier(setting)

    try:
        assert args['mode'] in ['train', 'evaluate', 'predict']
    except AssertionError:
        print('The option \'mode\' ust be train|evaluate|predict.')
        sys.exit(0)

    task_mode = getattr(mnist, args['mode'])
    task_mode()
