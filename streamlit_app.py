import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pywt
import cv2
from tensorflow.keras.layers import Layer

# Set page config
st.set_page_config(
    page_title="Ultrasound Breast Cancer Classification", 
    page_icon="🩺",
    layout="wide"
)

# Constants
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1

# Class names สำหรับ ultrasound breast cancer (2 classes)
CLASS_NAMES = [
    "Benign",
    "Malignant"
]

# Custom AdvancedLearnableEntropyPooling2D Layer (จากโมเดลจริงของคุณ)
class AdvancedLearnableEntropyPooling2D(Layer):
    """
    Advanced Learnable Entropy Pooling Layer for medical image classification
    """
    
    def __init__(self, pool_size=(2, 2), strides=(2, 2), epsilon=1e-06, learnable_mode='full', **kwargs):
        super(AdvancedLearnableEntropyPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.epsilon = epsilon
        self.learnable_mode = learnable_mode
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # Initialize learnable parameters based on mode
        if self.learnable_mode == 'full':
            # Learnable entropy weights for each channel
            self.entropy_weights = self.add_weight(
                name='entropy_weights',
                shape=(self.channels,),
                initializer='ones',
                trainable=True
            )
            
            # Learnable pooling bias
            self.pooling_bias = self.add_weight(
                name='pooling_bias',
                shape=(self.channels,),
                initializer='zeros',
                trainable=True
            )
        
        super(AdvancedLearnableEntropyPooling2D, self).build(input_shape)
    
    def call(self, inputs):
        """Apply entropy-based pooling with learnable parameters"""
        try:
            # Get input shape
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            channels = tf.shape(inputs)[3]
            
            # Calculate output dimensions
            out_height = (height - self.pool_size[0]) // self.strides[0] + 1
            out_width = (width - self.pool_size[1]) // self.strides[1] + 1
            
            # Initialize output tensor
            outputs = []
            
            # Apply pooling for each position
            for i in range(0, height - self.pool_size[0] + 1, self.strides[0]):
                row_outputs = []
                for j in range(0, width - self.pool_size[1] + 1, self.strides[1]):
                    # Extract pooling window
                    window = inputs[:, i:i+self.pool_size[0], j:j+self.pool_size[1], :]
                    
                    # Calculate entropy-based pooling
                    # Normalize window values to probabilities
                    window_normalized = tf.nn.softmax(tf.reshape(window, [batch_size, -1, channels]), axis=1)
                    
                    # Calculate entropy
                    entropy = -tf.reduce_sum(
                        window_normalized * tf.math.log(window_normalized + self.epsilon), 
                        axis=1
                    )
                    
                    # Apply learnable weights if in full mode
                    if self.learnable_mode == 'full':
                        entropy = entropy * self.entropy_weights + self.pooling_bias
                    
                    row_outputs.append(entropy)
                
                if row_outputs:
                    outputs.append(tf.stack(row_outputs, axis=1))
            
            if outputs:
                pooled_output = tf.stack(outputs, axis=1)
            else:
                # Fallback to average pooling if entropy calculation fails
                pooled_output = tf.nn.avg_pool2d(
                    inputs,
                    ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                    strides=[1, self.strides[0], self.strides[1], 1],
                    padding='VALID'
                )
            
            return pooled_output
            
        except Exception as e:
            # Fallback to average pooling in case of errors
            st.warning(f"Entropy pooling fallback activated: {e}")
            return tf.nn.avg_pool2d(
                inputs,
                ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                strides=[1, self.strides[0], self.strides[1], 1],
                padding='VALID'
            )
    
    def compute_output_shape(self, input_shape):
        def calc_output_size(input_size, pool_size, stride):
            if input_size is None:
                return None
            return (input_size - pool_size) // stride + 1
        
        return (input_shape[0],
                calc_output_size(input_shape[1], self.pool_size[0], self.strides[0]),
                calc_output_size(input_shape[2], self.pool_size[1], self.strides[1]),
                input_shape[3])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "epsilon": self.epsilon,
            "learnable_mode": self.learnable_mode
        })
        return config

# Preprocessing function สำหรับ Symlet2
def apply_symlet2_preprocessing(image_array):
    """
    Apply Symlet2 wavelet preprocessing to ultrasound image
    """
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Ensure single channel
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        
        # Normalize to [0, 1]
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        
        processed_image = np.zeros_like(image_array, dtype=np.float32)
        
        for c in range(image_array.shape[2]):
            channel_data = image_array[:, :, c]
            
            try:
                # Single-level DWT with sym2
                coeffs = pywt.dwt2(channel_data, 'sym2', mode='symmetric')
                cA, (cH, cV, cD) = coeffs
                
                # Enhanced reconstruction - boost important features
                enhanced_cA = cA * 1.05
                enhanced_cH = cH * 1.02  # Horizontal edges
                enhanced_cV = cV * 1.02  # Vertical edges
                enhanced_cD = cD * 1.01  # Diagonal features
                
                # Reconstruct with enhanced coefficients
                reconstructed = pywt.idwt2(
                    (enhanced_cA, (enhanced_cH, enhanced_cV, enhanced_cD)),
                    'sym2',
                    mode='symmetric'
                )
                
                # Ensure target shape
                target_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
                if reconstructed.shape != target_shape:
                    if reconstructed.shape[0] > target_shape[0] or reconstructed.shape[1] > target_shape[1]:
                        reconstructed = reconstructed[:target_shape[0], :target_shape[1]]
                    else:
                        pad_h = max(0, target_shape[0] - reconstructed.shape[0])
                        pad_w = max(0, target_shape[1] - reconstructed.shape[1])
                        if pad_h > 0 or pad_w > 0:
                            reconstructed = np.pad(reconstructed,
                                                 ((0, pad_h), (0, pad_w)),
                                                 mode='edge')
                
                processed_image[:, :, c] = np.clip(reconstructed, 0, 1)
                
            except Exception as e:
                st.warning(f"Wavelet processing failed: {e}")
                enhanced_original = channel_data * 1.02
                processed_image[:, :, c] = np.clip(enhanced_original, 0, 1)
        
        return processed_image.astype(np.float32)
    
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return image_array

@st.cache_resource
def load_model():
    """Load the trained model with custom objects"""
    
    # Define custom objects
    custom_objects = {
        'AdvancedLearnableEntropyPooling2D': AdvancedLearnableEntropyPooling2D
    }
    
    # Try to load different model files
    model_files = [
        'symmrnet_symlet2_3blocks.h5',
        'ultrasound_model_vmc_net.h5'
    ]
    
    for model_file in model_files:
        try:
            st.info(f"🔄 Trying to load: {model_file}")
            model = tf.keras.models.load_model(
                model_file, 
                custom_objects=custom_objects,
                compile=False  # Skip compilation to avoid optimizer issues
            )
            st.success(f"✅ Successfully loaded: {model_file}")
            return model, model_file
            
        except FileNotFoundError:
            st.warning(f"⚠️ File not found: {model_file}")
            continue
        except Exception as e:
            st.error(f"❌ Error loading {model_file}: {str(e)}")
            continue
    
    return None, None

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    try:
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Resize to model input size
        if len(image_array.shape) == 3:
            image_resized = cv2.resize(image_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
        else:
            image_resized = cv2.resize(image_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Apply Symlet2 preprocessing
        processed_image = apply_symlet2_preprocessing(image_resized)
        
        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)
        
        return processed_image
    
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

def predict_image(image_array, model):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(image_array, verbose=0)
        
        # Handle different output formats
        if len(predictions.shape) > 1 and predictions.shape[1] >= 2:
            # Binary classification with 2 outputs
            benign_prob = float(predictions[0][0])
            malignant_prob = float(predictions[0][1])
        else:
            # Single output (sigmoid)
            malignant_prob = float(predictions[0][0] if len(predictions.shape) > 1 else predictions[0])
            benign_prob = 1.0 - malignant_prob
        
        results = [
            {'class': 'Benign', 'probability': benign_prob},
            {'class': 'Malignant', 'probability': malignant_prob}
        ]
        
        # Sort by probability (highest first)
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        return results
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Main Streamlit App
def main():
    st.title("🩺 Ultrasound Breast Cancer Classification")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a SymMRNet model with Advanced Learnable Entropy Pooling 
        to classify ultrasound breast images as **Benign** or **Malignant**.
        """)
        
        st.header("📋 Instructions")
        st.write("""
        1. Upload an ultrasound breast image
        2. Click 'Analyze Image' button
        3. View the classification results
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for research purposes only. 
        Always consult with healthcare professionals 
        for medical diagnosis.
        """)
    
    # Load model
    model, model_name = load_model()
    if model is None:
        st.error("❌ Cannot load any model. Please check if model files exist in the repository.")
        st.info("Expected files: symmrnet_symlet2_3blocks.h5 or ultrasound_model_vmc_net.h5")
        st.stop()
    else:
        st.success(f"✅ Model loaded successfully: {model_name}")
        
        # Display model info
        with st.expander("🔍 Model Information"):
            st.write(f"**Model file:** {model_name}")
            st.write(f"**Input shape:** {model.input_shape}")
            st.write(f"**Output shape:** {model.output_shape}")
            st.write(f"**Total parameters:** {model.count_params():,}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an ultrasound breast image...", 
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a grayscale or color ultrasound image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
    
    with col2:
        st.header("🔬 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        results = predict_image(processed_image, model)
                        
                        if results is not None:
                            st.success("✅ Analysis Complete!")
                            
                            # Display main prediction
                            main_prediction = results[0]
                            confidence = main_prediction['probability']
                            
                            if main_prediction['class'] == 'Malignant':
                                st.error(f"🔴 **{main_prediction['class']}** ({confidence:.1%} confidence)")
                            else:
                                st.success(f"🟢 **{main_prediction['class']}** ({confidence:.1%} confidence)")
                            
                            # Display detailed results
                            st.subheader("📊 Detailed Probabilities")
                            for result in results:
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.write(f"**{result['class']}**")
                                with col_b:
                                    st.write(f"{result['probability']:.1%}")
                                
                                # Progress bar
                                st.progress(float(result['probability']))
                            
                            # Additional visualization
                            st.subheader("📈 Confidence Chart")
                            chart_data = {
                                'Classification': [r['class'] for r in results],
                                'Probability': [r['probability'] for r in results]
                            }
                            st.bar_chart(chart_data, x='Classification', y='Probability')
        else:
            st.info("👆 Please upload an image to start analysis")

if __name__ == "__main__":
    main()
