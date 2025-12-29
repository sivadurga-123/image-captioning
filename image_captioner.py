import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

class ImageCaptioner:
    """Image captioning model using CNN-RNN architecture"""
    
    def __init__(self, vocab_size=5000, max_length=40, embedding_dim=128, lstm_units=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.feature_extractor = None
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def load_feature_extractor(self, model_type='vgg16'):
        """Load pre-trained CNN for feature extraction"""
        if model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            raise ValueError('Model type should be vgg16 or resnet50')
        
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        # Freeze the feature extractor weights
        self.feature_extractor.trainable = False
        return self.feature_extractor
    
    def build_model(self):
        """Build the encoder-decoder model"""
        # Encoder
        image_input = tf.keras.Input(shape=(7, 7, 512))  # VGG16 output shape
        encoder = LSTM(self.lstm_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(image_input)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_input = tf.keras.Input(shape=(None,))
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(decoder_input)
        decoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.model = Model([image_input, decoder_input], decoder_outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        features = self.feature_extractor.predict(img_array)
        return features
    
    def generate_caption(self, image_path, max_length=40):
        """Generate caption for an image"""
        features = self.extract_features(image_path)
        
        # Start with the start token
        in_text = '<START>'
        for i in range(max_length):
            # Convert text to sequence
            sequence = [self.word_to_idx.get(word, 0) for word in in_text.split()]
            sequence = np.array(sequence).reshape(1, -1)
            
            # Predict next word
            yhat = self.model.predict([features, sequence], verbose=0)
            yhat = np.argmax(yhat, axis=-1)[0][-1]
            word = self.idx_to_word.get(yhat, '<UNK>')
            
            in_text += ' ' + word
            if word == '<END>':
                break
        
        return in_text
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model

def main():
    print('Image Captioning System')
    print('Using VGG16 for feature extraction and LSTM for caption generation')
    
    # Initialize captioner
    captioner = ImageCaptioner(vocab_size=5000, max_length=40, embedding_dim=128, lstm_units=256)
    captioner.load_feature_extractor('vgg16')
    captioner.build_model()
    
    # Generate sample captions
    print('Model initialized and ready for training.')

if __name__ == '__main__':
    main()
