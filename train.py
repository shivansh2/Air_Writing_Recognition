import tensorflow as tf
from model import build_model

def train_model(train_data, val_data, model_save_path):
    train_images, train_labels = train_data['data'], train_data['labels']
    val_images, val_labels = val_data['data'], val_data['labels']

    model = build_model(input_shape=train_images[0].shape, num_classes=len(np.unique(train_labels)))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels,
              validation_data=(val_images, val_labels),
              epochs=10,
              batch_size=32)
    
    model.save(model_save_path)
