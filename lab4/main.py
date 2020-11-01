from model import *
from data import *

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20, 'data/hiragana/train', 'image', 'label', data_gen_args, save_to_dir="data/hiragana/train/aug")

num_batch = 3
for i, batch in enumerate(myGenerator):
    if(i >= num_batch):
        break

image_arr, mask_arr = geneTrainNpy("data/hiragana/train/aug/", "data/hiragana/train/aug/")

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2, 'data/hiragana/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_hiragana.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.compile(loss=dice_coef_loss, optimizer=Adam(lr=3e-5), metrics=[dice_coef, 'accuracy'])
model.fit_generator(myGene, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

testGene = testGenerator("data/hiragana/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/hiragana/test", results)