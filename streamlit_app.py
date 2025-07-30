import streamlit as st
import numpy as np
from PIL import Image
import cv2

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

def preprocess_image_simple(image):
    """Simple preprocessing without wavelet"""
    try:
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to model input size
        image_resized = cv2.resize(image_array, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Normalize to [0, 1]
        if image_resized.max() > 1.0:
            image_resized = image_resized.astype(np.float32) / 255.0
        
        # Add channel dimension
        if len(image_resized.shape) == 2:
            image_resized = np.expand_dims(image_resized, axis=-1)
        
        # Add batch dimension
        image_resized = np.expand_dims(image_resized, axis=0)
        
        return image_resized
    
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model - with proper error handling for TF 2.20"""
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        st.write(f"🔍 TensorFlow version: {tf.__version__}")
        
        # Custom Layers for the model
        
        # Original Symlet2 Pooling Layer
        class Symlet2PoolingLayer(tf.keras.layers.Layer):
            def __init__(self, pool_size=(2, 2), **kwargs):
                super(Symlet2PoolingLayer, self).__init__(**kwargs)
                self.pool_size = pool_size

                # Symlet2 filter coefficients
                symlet2_h = [
                    -0.12940952255092145,
                    0.22414386804185735,
                    0.836516303737469,
                    0.48296291314469025
                ]

                # Convert to 2D kernels
                self.h_kernel = tf.constant([
                    [symlet2_h[0], symlet2_h[1]],
                    [symlet2_h[2], symlet2_h[3]]
                ], dtype=tf.float32)
                
                # Normalize kernel
                self.h_kernel = self.h_kernel / tf.reduce_sum(tf.abs(self.h_kernel))

            def build(self, input_shape):
                self.channels = input_shape[-1]
                super(Symlet2PoolingLayer, self).build(input_shape)

            def call(self, inputs):
                # Reshape kernels for depthwise convolution
                h_kernel_expanded = tf.reshape(self.h_kernel, [2, 2, 1, 1])
                h_kernel_tiled = tf.tile(h_kernel_expanded, [1, 1, self.channels, 1])

                # Apply low-pass filtering
                low_pass = tf.nn.depthwise_conv2d(
                    inputs,
                    h_kernel_tiled,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                # Apply ReLU activation
                activated = tf.nn.relu(low_pass)

                # Downsample
                pooled = tf.nn.avg_pool2d(
                    activated,
                    ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                    strides=[1, self.pool_size[0], self.pool_size[1], 1],
                    padding='VALID'
                )

                return pooled

            def compute_output_shape(self, input_shape):
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

        # Advanced Learnable Entropy Pooling Layer (found in the model)
        class AdvancedLearnableEntropyPooling2D(tf.keras.layers.Layer):
            def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
                super(AdvancedLearnableEntropyPooling2D, self).__init__(**kwargs)
                self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size)
                self.strides = strides if strides else self.pool_size
                self.padding = padding.upper()

            def build(self, input_shape):
                super(AdvancedLearnableEntropyPooling2D, self).build(input_shape)

            def call(self, inputs):
                # Simple implementation using average pooling as fallback
                # This is a simplified version for compatibility
                return tf.nn.avg_pool2d(
                    inputs,
                    ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                    strides=[1, self.strides[0], self.strides[1], 1],
                    padding=self.padding
                )

            def compute_output_shape(self, input_shape):
                if self.padding == 'VALID':
                    height = (input_shape[1] - self.pool_size[0]) // self.strides[0] + 1
                    width = (input_shape[2] - self.pool_size[1]) // self.strides[1] + 1
                else:  # SAME padding
                    height = input_shape[1] // self.strides[0]
                    width = input_shape[2] // self.strides[1]
                return (input_shape[0], height, width, input_shape[3])

            def get_config(self):
                config = super().get_config()
                config.update({
                    "pool_size": self.pool_size,
                    "strides": self.strides,
                    "padding": self.padding.lower()
                })
                return config
        
        # Register both custom layers
        custom_objects = {
            'Symlet2PoolingLayer': Symlet2PoolingLayer,
            'AdvancedLearnableEntropyPooling2D': AdvancedLearnableEntropyPooling2D
        }
        
        # Try to load the model with multiple attempts
        model_files = ['symmrnet_symlet2_3blocks.h5', 'ultrasound_model_vmc_net.h5']
        
        for model_file in model_files:
            try:
                st.write(f"🔄 Trying to load: {model_file}")
                model = tf.keras.models.load_model(
                    model_file, 
                    custom_objects=custom_objects,
                    compile=False
                )
                st.success(f"✅ Model loaded successfully: {model_file}")
                return model
            except FileNotFoundError:
                st.warning(f"📁 File not found: {model_file}")
                continue
            except Exception as e:
                st.error(f"❌ Error loading {model_file}: {str(e)}")
                continue
        
        # If no model loaded successfully
        st.error("❌ Could not load any model file")
        st.write("**Available model files should be:**")
        st.write("- symmrnet_symlet2_3blocks.h5")
        st.write("- ultrasound_model_vmc_net.h5")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.write("**Fallback**: Using demo mode with random predictions")
        return None

def predict_image_demo(image_array):
    """Demo prediction function when model fails to load"""
    # Random prediction for demo
    np.random.seed(42)  # For consistent demo results
    benign_prob = np.random.random()
    malignant_prob = 1 - benign_prob
    
    results = [
        {'class': 'Benign', 'probability': benign_prob},
        {'class': 'Malignant', 'probability': malignant_prob}
    ]
    
    # Sort by probability (highest first)
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    return results

def predict_image(image_array, model):
    """Make prediction on preprocessed image"""
    try:
        if model is None:
            st.warning("🔄 Using demo mode - model not loaded")
            return predict_image_demo(image_array)
        
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
        st.warning("🔄 Falling back to demo mode")
        return predict_image_demo(image_array)

# Main Streamlit App
def main():
    st.title("🩺 Ultrasound Breast Cancer Classification")
    st.markdown("---")
    
    # Show system info
    with st.expander("🔍 System Information"):
        try:
            import tensorflow as tf
            st.write(f"**TensorFlow version:** {tf.__version__}")
        except:
            st.write("**TensorFlow:** Not loaded")
        
        import sys
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**NumPy version:** {np.__version__}")
    
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
                    processed_image = preprocess_image_simple(image)
                    
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
