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

# Custom Symlet2 Pooling Layer (จากโมเดลของคุณ)
class Symlet2PoolingLayer(Layer):
    """
    Pure TensorFlow implementation of Symlet2 wavelet pooling.
    FIXED: Uses VALID padding to match MaxPooling2D behavior exactly.
    """

    def __init__(self, pool_size=(2, 2), **kwargs):
        super(Symlet2PoolingLayer, self).__init__(**kwargs)
        self.pool_size = pool_size

        # Actual Symlet2 filter coefficients (h0, h1, h2, h3)
        # Low-pass filter (scaling function)
        symlet2_h = [
            -0.12940952255092145,
            0.22414386804185735,
            0.836516303737469,
            0.48296291314469025
        ]

        # High-pass filter (wavelet function)
        symlet2_g = [
            -0.48296291314469025,
            0.836516303737469,
            -0.22414386804185735,
            -0.12940952255092145
        ]

        # Convert to 2D kernels for convolution (2x2 approximation)
        # Take every other coefficient to form 2x2 kernel
        self.h_kernel = tf.constant([
            [symlet2_h[0], symlet2_h[1]],
            [symlet2_h[2], symlet2_h[3]]
        ], dtype=tf.float32)

        self.g_kernel = tf.constant([
            [symlet2_g[0], symlet2_g[1]],
            [symlet2_g[2], symlet2_g[3]]
        ], dtype=tf.float32)

        # Normalize kernels
        self.h_kernel = self.h_kernel / tf.reduce_sum(tf.abs(self.h_kernel))
        self.g_kernel = self.g_kernel / tf.reduce_sum(tf.abs(self.g_kernel))

    def build(self, input_shape):
        self.channels = input_shape[-1]
        super(Symlet2PoolingLayer, self).build(input_shape)

    def call(self, inputs):
        """Apply Symlet2 wavelet-based pooling operation."""
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Reshape kernels for depthwise convolution [filter_height, filter_width, in_channels, channel_multiplier]
        h_kernel_expanded = tf.reshape(self.h_kernel, [2, 2, 1, 1])
        h_kernel_tiled = tf.tile(h_kernel_expanded, [1, 1, self.channels, 1])

        # Apply low-pass filtering (approximation coefficients)
        low_pass = tf.nn.depthwise_conv2d(
            inputs,
            h_kernel_tiled,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        # Apply ReLU activation for non-linearity
        activated = tf.nn.relu(low_pass)

        # Downsample by factor of 2 (equivalent to keeping LL subband)
        # Use VALID padding to match MaxPooling2D behavior exactly
        pooled = tf.nn.avg_pool2d(
            activated,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.pool_size[0], self.pool_size[1], 1],
            padding='VALID'
        )

        return pooled

    def compute_output_shape(self, input_shape):
        # Calculate output shape for VALID padding (same as MaxPooling2D)
        # Formula: (input_size - pool_size) // stride + 1
        def calc_output_size(input_size, pool_size, stride):
            if input_size is None:
                return None
            return (input_size - pool_size) // stride + 1

        return (input_shape[0],
                calc_output_size(input_shape[1], self.pool_size[0], self.pool_size[0]),
                calc_output_size(input_shape[2], self.pool_size[1], self.pool_size[1]),
                input_shape[3])

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
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
    try:
        # Register custom layer
        custom_objects = {'Symlet2PoolingLayer': Symlet2PoolingLayer}
        model = tf.keras.models.load_model(
            'symmrnet_symlet2_3blocks.h5', 
            custom_objects=custom_objects,
            compile=False  # ป้องกันปัญหา optimizer compatibility
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.write("ปัญหาที่พบ:", str(e))
        return None

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
        
        # Get prediction probabilities
        benign_prob = float(predictions[0][0])
        malignant_prob = float(predictions[0][1])
        
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
        This application uses a SymMRNet model with Symlet2 wavelet pooling 
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
    model = load_model()
    if model is None:
        st.error("❌ Cannot load model. Please check if 'symmrnet_symlet2_3blocks.h5' exists.")
        st.stop()
    else:
        st.success("✅ Model loaded successfully!")
    
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
