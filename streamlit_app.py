import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pywt
import cv2
from tensorflow.keras.layers import Layer

# Set page config
st.set_page_config(
    page_title="Ultrasound Breast Cancer Classification by MIDTHaI", 
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

# Custom AdvancedLearnableEntropyPooling2D Layer (ตาม architecture จริงของคุณ)
class AdvancedLearnableEntropyPooling2D(Layer):
    """
    🧠 Advanced Learnable Entropy-Guided Pooling - EXACT COPY from your training code
    Learnable Implementation of All 3 Steps from Mathematical Formulation
    """

    def __init__(self, pool_size=(2, 2), strides=None, epsilon=1e-6,
                 learnable_mode='full', **kwargs):
        super(AdvancedLearnableEntropyPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.epsilon = epsilon
        self.learnable_mode = learnable_mode

    def build(self, input_shape):
        super(AdvancedLearnableEntropyPooling2D, self).build(input_shape)

        if self.learnable_mode in ['entropy_only', 'full']:
            # 📊 Step 1: Learnable Entropy Calculation Parameters
            self.learnable_epsilon = self.add_weight(
                name='learnable_epsilon',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(self.epsilon),
                constraint=tf.keras.constraints.NonNeg(),
                trainable=True
            )

            self.prob_scale = self.add_weight(
                name='prob_scale',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                constraint=tf.keras.constraints.NonNeg(),
                trainable=True
            )

            self.entropy_scale = self.add_weight(
                name='entropy_scale',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )

        if self.learnable_mode in ['entropy_only', 'full']:
            # 📊 Step 1→2: Learnable Entropy Transformation
            self.entropy_weight = self.add_weight(
                name='entropy_weight',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )
            self.entropy_bias = self.add_weight(
                name='entropy_bias',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True
            )

        if self.learnable_mode in ['alpha_only', 'full']:
            # ⚖️ Step 2: Learnable Alpha Adjustment
            self.alpha_scale = self.add_weight(
                name='alpha_scale',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )
            self.alpha_shift = self.add_weight(
                name='alpha_shift',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True
            )

        if self.learnable_mode == 'full':
            # 🎯 Step 3: Learnable Max/Avg Preference
            self.max_preference = self.add_weight(
                name='max_preference',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(0.5),
                trainable=True
            )

            self.max_weight = self.add_weight(
                name='max_weight',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )
            self.avg_weight = self.add_weight(
                name='avg_weight',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )

        if self.learnable_mode == 'attention_based':
            # 🔍 Attention-based learnable mechanism
            self.attention_dense = tf.keras.layers.Dense(
                1, activation='sigmoid', name='entropy_attention'
            )

    def call(self, inputs):
        # Extract patches using TensorFlow's efficient operation
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        batch_size = tf.shape(patches)[0]
        output_height = tf.shape(patches)[1]
        output_width = tf.shape(patches)[2]
        channels = tf.shape(inputs)[3]
        patch_size = self.pool_size[0] * self.pool_size[1]

        patches_reshaped = tf.reshape(patches,
                                     [batch_size, output_height, output_width, channels, patch_size])

        # Compute Max and Average pooling
        max_pooled = tf.reduce_max(patches_reshaped, axis=-1)
        avg_pooled = tf.reduce_mean(patches_reshaped, axis=-1)

        # 📊 Step 1: Learnable Shannon Entropy Calculation
        if self.learnable_mode in ['entropy_only', 'full']:
            entropies = self._learnable_entropy_calculation(patches_reshaped)
        else:
            # Standard entropy calculation
            patches_positive = patches_reshaped + self.epsilon
            patches_sum = tf.reduce_sum(patches_positive, axis=-1, keepdims=True)
            patches_sum = tf.maximum(patches_sum, self.epsilon)
            probabilities = patches_positive / patches_sum

            log_probs = tf.math.log(tf.maximum(probabilities, self.epsilon))
            entropies = -tf.reduce_sum(probabilities * log_probs, axis=-1)

        # Apply different strategies for alpha calculation
        if self.learnable_mode == 'entropy_only':
            alpha = self._learnable_entropy_alpha(entropies)
        elif self.learnable_mode == 'alpha_only':
            alpha = self._learnable_alpha_adjustment(entropies)
        elif self.learnable_mode == 'full':
            alpha = self._full_learnable_alpha(entropies)
        elif self.learnable_mode == 'attention_based':
            alpha = self._attention_based_alpha(entropies, patches_reshaped)
        else:
            alpha = self._basic_learnable_alpha(entropies)

        # 🎯 Step 3: Learnable Hybrid Pooling
        if self.learnable_mode == 'full' and hasattr(self, 'max_preference'):
            enhanced_max = self.max_weight * max_pooled
            enhanced_avg = self.avg_weight * avg_pooled
            adjusted_alpha = alpha * self.max_preference + (1 - self.max_preference) * 0.5
            output = adjusted_alpha * enhanced_max + (1 - adjusted_alpha) * enhanced_avg
        else:
            output = alpha * max_pooled + (1 - alpha) * avg_pooled

        # Safety checks
        output = tf.where(tf.math.is_nan(output), avg_pooled, output)
        output = tf.where(tf.math.is_inf(output), max_pooled, output)

        return output

    def _learnable_entropy_calculation(self, patches_reshaped):
        """📊 Step 1: Learnable Enhancement of Shannon Entropy"""
        effective_epsilon = tf.maximum(self.learnable_epsilon, 1e-8)
        patches_scaled = patches_reshaped * self.prob_scale + effective_epsilon
        patches_sum = tf.reduce_sum(patches_scaled, axis=-1, keepdims=True)
        patches_sum = tf.maximum(patches_sum, effective_epsilon)
        probabilities = patches_scaled / patches_sum

        log_probs = tf.math.log(tf.maximum(probabilities, effective_epsilon))
        raw_entropy = -tf.reduce_sum(probabilities * log_probs, axis=-1)
        entropies = self.entropy_scale * raw_entropy
        return entropies

    def _learnable_entropy_alpha(self, entropies):
        """📊 Learnable entropy transformation only"""
        h_mean = tf.reduce_mean(entropies, axis=(1, 2), keepdims=True)
        h_std = tf.math.reduce_std(entropies, axis=(1, 2), keepdims=True)
        h_std = tf.maximum(h_std, self.epsilon)
        normalized_entropy = (entropies - h_mean) / h_std
        alpha = tf.nn.sigmoid(self.entropy_weight * normalized_entropy + self.entropy_bias)
        return tf.clip_by_value(alpha, 0.0, 1.0)

    def _learnable_alpha_adjustment(self, entropies):
        """⚖️ Original alpha with learnable adjustment"""
        h_min = tf.reduce_min(entropies, axis=(1, 2), keepdims=True)
        h_max = tf.reduce_max(entropies, axis=(1, 2), keepdims=True)
        h_range = tf.maximum(h_max - h_min, self.epsilon)
        alpha_base = (entropies - h_min) / h_range
        alpha = self.alpha_scale * alpha_base + self.alpha_shift
        return tf.clip_by_value(alpha, 0.0, 1.0)

    def _full_learnable_alpha(self, entropies):
        """🧠 Step 2: Enhanced Alpha Formula"""
        h_min = tf.reduce_min(entropies, axis=(1, 2), keepdims=True)
        h_max = tf.reduce_max(entropies, axis=(1, 2), keepdims=True)
        
        effective_epsilon = tf.maximum(self.learnable_epsilon, 1e-8) if hasattr(self, 'learnable_epsilon') else self.epsilon
        h_range = tf.maximum(h_max - h_min, effective_epsilon)
        alpha_original = (entropies - h_min) / h_range
        
        alpha_enhanced = self.entropy_weight * alpha_original + self.entropy_bias
        alpha_sigmoid = tf.nn.sigmoid(alpha_enhanced)
        alpha_final = self.alpha_scale * alpha_sigmoid + self.alpha_shift
        return tf.clip_by_value(alpha_final, 0.0, 1.0)

    def _attention_based_alpha(self, entropies, patches):
        """🔍 Attention mechanism for alpha calculation"""
        patch_max = tf.reduce_max(patches, axis=-1)
        patch_mean = tf.reduce_mean(patches, axis=-1)
        patch_std = tf.math.reduce_std(patches, axis=-1)
        features = tf.stack([entropies, patch_max, patch_mean, patch_std], axis=-1)
        alpha = self.attention_dense(features)
        alpha = tf.squeeze(alpha, axis=-1)
        return tf.clip_by_value(alpha, 0.0, 1.0)

    def _basic_learnable_alpha(self, entropies):
        """Basic learnable version"""
        h_mean = tf.reduce_mean(entropies, axis=(1, 2), keepdims=True)
        h_std = tf.math.reduce_std(entropies, axis=(1, 2), keepdims=True)
        h_std = tf.maximum(h_std, self.epsilon)
        normalized_entropy = (entropies - h_mean) / h_std
        alpha = tf.nn.sigmoid(normalized_entropy)
        return alpha

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        output_height = (height - self.pool_size[0]) // self.strides[0] + 1
        output_width = (width - self.pool_size[1]) // self.strides[1] + 1
        return (batch_size, output_height, output_width, channels)

    def get_config(self):
        config = super(AdvancedLearnableEntropyPooling2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'epsilon': self.epsilon,
            'learnable_mode': self.learnable_mode
        })
        return config

# Preprocessing function (เหมือนเดิม - ไม่ใช้ Symlet2 เพราะโมเดลไม่ได้ใช้)
def preprocess_image_for_vmc_net(image_array):
    """
    Simple preprocessing for VMC-Net (normalize only)
    Based on your training code: x / 255.0
    """
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Ensure single channel
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        
        # Simple normalization (ตามโค้ดต้นฉบับ: x / 255.0)
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        
        return image_array.astype(np.float32)
    
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

@st.cache_resource
def load_model():
    """Load model with comprehensive fallback strategies"""
    
    custom_objects = {
        'AdvancedLearnableEntropyPooling2D': AdvancedLearnableEntropyPooling2D
    }
    
    model_files = [
        'ultrasound_model_vmc_net.h5',  # ลองไฟล์นี้ก่อน (VMC-Net)
        'symmrnet_symlet2_3blocks.h5'   # ไฟล์สำรอง
    ]
    
    for model_file in model_files:
        st.info(f"🔄 Trying to load: {model_file}")
        
        # Strategy 1: Load with custom objects
        try:
            model = tf.keras.models.load_model(
                model_file, 
                custom_objects=custom_objects,
                compile=False
            )
            st.success(f"✅ Strategy 1 Success: {model_file}")
            return model, model_file
        except Exception as e1:
            st.warning(f"Strategy 1 failed: {str(e1)[:80]}...")
        
        # Strategy 2: Load weights only (rebuild architecture)
        try:
            # สร้าง architecture ใหม่แล้วโหลด weights
            model = build_vmc_net_architecture()
            model.load_weights(model_file)
            st.success(f"✅ Strategy 2 Success (weights only): {model_file}")
            return model, model_file
        except Exception as e2:
            st.warning(f"Strategy 2 failed: {str(e2)[:80]}...")
            
        # Strategy 3: Safe loading (skip custom layers)
        try:
            model = tf.keras.models.load_model(model_file, compile=False)
            st.info(f"⚠️ Strategy 3 (partial load): {model_file}")
            return model, model_file
        except Exception as e3:
            st.warning(f"Strategy 3 failed: {str(e3)[:80]}...")
    
    return None, None

def build_vmc_net_architecture():
    """Build VMC-Net architecture exactly as in training code"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, AveragePooling2D
    from tensorflow.keras.optimizers import Adam
    
    NUM_CLASSES = 2
    LEARNING_RATE = 0.0003
    
    model = Sequential([
        # First convolutional block
        Conv2D(8, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
               padding='same', name='conv2d'),  # Output: 64x64x8
        AdvancedLearnableEntropyPooling2D(pool_size=(2, 2), learnable_mode='full', name='entropy_pooling_1'),  # Output: 32x32x8

        # Second convolutional block
        Conv2D(16, (2, 2), activation='relu', padding='valid', name='conv2d_1'),  # Output: 31x31x16
        AdvancedLearnableEntropyPooling2D(pool_size=(2, 2), learnable_mode='full', name='entropy_pooling_2'),  # Output: 15x15x16

        # Third convolutional block
        Conv2D(64, (2, 2), activation='relu', padding='valid', name='conv2d_2'),  # Output: 14x14x64
        AdvancedLearnableEntropyPooling2D(pool_size=(2, 2), learnable_mode='full', name='entropy_pooling_3'),  # Output: 7x7x64

        # Fourth convolutional block
        Conv2D(112, (3, 3), activation='relu', padding='valid', name='conv2d_3'),  # Output: 5x5x112
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='average_pooling2d'),  # Output: 2x2x112

        # Flatten and dense layers
        Flatten(name='flatten'),  # Output: 448
        Dropout(0.5, name='dropout'),  # Output: 448
        Dense(192, activation='relu', name='dense'),  # Output: 192
        Dropout(0.5, name='dropout_1'),  # Output: 192
        Dense(NUM_CLASSES, activation='softmax', name='dense_1')  # Output: 2
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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
        
        # Apply VMC-Net preprocessing (simple normalization)
        processed_image = preprocess_image_for_vmc_net(image_resized)
        
        if processed_image is not None:
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
            # Binary classification with 2 outputs (softmax)
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
        st.header("ℹ️ About MIDTHaI")
        st.write("""
        This application uses a Entropy Weighted Hybrid Pooling model in CNN
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
        st.info("**Troubleshooting:**")
        st.info("1. Make sure model files are uploaded to GitHub")
        st.info("2. Check file names are exactly: `ultrasound_model_vmc_net.h5` or `symmrnet_symlet2_3blocks.h5`")
        st.info("3. Verify file size is under 100MB (your file is 1.9 MB - should be OK)")
        
        # Additional help
        st.header("🔧 Debug Information")
        st.code("""
        Expected model architecture (VMC-Net):
        - 4 Conv2D blocks
        - 3 AdvancedLearnableEntropyPooling2D layers  
        - learnable_mode='full'
        - Input: (64, 64, 1)
        - Output: 2 classes (Benign/Malignant)
        """)
        st.stop()
    else:
        st.success(f"✅ Model loaded successfully: {model_name}")
        
        # Display model info
        with st.expander("🔍 Model Information"):
            try:
                st.write(f"**Model file:** {model_name}")
                st.write(f"**Architecture:** VMC-Net with Advanced Learnable Entropy Pooling")
                st.write(f"**Input shape:** {model.input_shape}")
                st.write(f"**Output shape:** {model.output_shape}")
                st.write(f"**Total parameters:** {model.count_params():,}")
                
                # Show layer summary
                st.subheader("Layer Information")
                for i, layer in enumerate(model.layers):
                    st.write(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
                    
            except Exception as e:
                st.write(f"**Model file:** {model_name}")
                st.write("Model information partially available")
    
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
        else:
            st.info("👆 Please upload an image to start analysis")

if __name__ == "__main__":
    main()
