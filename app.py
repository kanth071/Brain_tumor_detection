import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import base64
import threading
import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_WEIGHTS_PATH = os.path.join(BASE_DIR, 'model_weights.npz')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'model.weights.h5')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
TARGET_LAYER_NAME = 'block5_conv4'
IMG_SIZE = (240, 240)

# Global Application Configuration
app_config = {
    'target_layer': 'block5_conv4',
    'colormap': 'JET',
    'alpha': 0.45,
    'auto_reload': True
}

# In-memory Reports History
reports_history = []
report_id_counter = 1001

COLORMAP_DICT = {
    'JET': cv2.COLORMAP_JET,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'HOT': cv2.COLORMAP_HOT,
    'INFERNO': cv2.COLORMAP_INFERNO,
    'PLASMA': cv2.COLORMAP_PLASMA
}

def create_fine_tuned_vgg19():
    """Builds the fine-tuned VGG19 architecture matching trained parameters."""
    base_model = VGG19(input_shape=(240, 240, 3), include_top=False, weights='imagenet')
    
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

class DynamicModelManager:
    def __init__(self, model_path, weights_path, npz_path):
        self.model_path = model_path
        self.weights_path = weights_path
        self.npz_path = npz_path
        self.model = None
        self.grad_models = {}
        self.last_mtime = 0
        self.lock = threading.Lock()
        self.load_model_if_updated()

    def build_grad_model_cache(self):
        """Pre-constructs Grad-CAM computational graphs for instant execution (< 0.5s)."""
        self.grad_models = {}
        if self.model is None:
            return
        target_layers = ['block5_conv4', 'block5_conv3', 'block5_conv2', 'block4_conv4']
        for layer_name in target_layers:
            try:
                target_layer = self.model.get_layer(layer_name)
                g_model = tf.keras.models.Model(
                    inputs=self.model.input,
                    outputs=[target_layer.output, self.model.output]
                )
                self.grad_models[layer_name] = g_model
            except Exception as e:
                print(f"[ModelManager] Skipping grad model for {layer_name}: {e}")

    def load_model_if_updated(self, force=False):
        if not app_config.get('auto_reload', True) and not force and self.model is not None:
            return True
        with self.lock:
            # 1. Prefer portable NumPy compressed weights file model_weights.npz (71MB - 100% version independent)
            if os.path.exists(self.npz_path):
                current_mtime = os.path.getmtime(self.npz_path)
                if current_mtime > self.last_mtime or self.model is None or force:
                    print(f"[ModelManager] Loading fine-tuned weights from {self.npz_path}...")
                    try:
                        built_model = create_fine_tuned_vgg19()
                        npz = np.load(self.npz_path)
                        weights = [npz[f'arr_{i}'] for i in range(len(npz.files))]
                        built_model.set_weights(weights)
                        self.model = built_model
                        self.last_mtime = current_mtime
                        self.build_grad_model_cache()
                        print("[ModelManager] Model & Grad-CAM cache successfully loaded from model_weights.npz.")
                        return True
                    except Exception as e:
                        print(f"[ModelManager] Error loading npz weights: {e}")

            # 2. Fallback to model.weights.h5
            if os.path.exists(self.weights_path):
                current_mtime = os.path.getmtime(self.weights_path)
                if current_mtime > self.last_mtime or self.model is None or force:
                    print(f"[ModelManager] Loading fine-tuned weights from {self.weights_path}...")
                    try:
                        built_model = create_fine_tuned_vgg19()
                        built_model.load_weights(self.weights_path)
                        self.model = built_model
                        self.last_mtime = current_mtime
                        self.build_grad_model_cache()
                        print("[ModelManager] Model & Grad-CAM cache successfully loaded from model.weights.h5.")
                        return True
                    except Exception as e:
                        print(f"[ModelManager] Error loading h5 weights: {e}")

            # 3. Fallback to full model.h5 if present locally
            if os.path.exists(self.model_path):
                current_mtime = os.path.getmtime(self.model_path)
                if current_mtime > self.last_mtime or self.model is None or force:
                    print(f"[ModelManager] Loading full model from {self.model_path}...")
                    try:
                        loaded_model = tf.keras.models.load_model(self.model_path)
                        self.model = loaded_model
                        self.last_mtime = current_mtime
                        self.build_grad_model_cache()
                        print("[ModelManager] Full model successfully loaded from model.h5.")
                        return True
                    except Exception as e:
                        print(f"[ModelManager] Error loading full model: {e}")

            if self.model is None:
                print("[ModelManager] Warning: No model weights file could be loaded.")
                return False
            return True

    def get_model(self):
        self.load_model_if_updated()
        return self.model, self.last_mtime

    def get_grad_model(self, layer_name):
        self.load_model_if_updated()
        if layer_name in self.grad_models:
            return self.grad_models[layer_name], layer_name
        if self.model:
            try:
                fallback_model = tf.keras.models.Model(
                    inputs=self.model.input,
                    outputs=[self.model.get_layer(layer_name).output, self.model.output]
                )
                return fallback_model, layer_name
            except Exception:
                pass
            conv_layers = [l.name for l in self.model.layers if 'conv' in l.name]
            fallback_name = conv_layers[-1] if conv_layers else 'block5_conv4'
            fallback_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(fallback_name).output, self.model.output]
            )
            return fallback_model, fallback_name
        return None, layer_name

model_manager = DynamicModelManager(MODEL_PATH, WEIGHTS_PATH, NPZ_WEIGHTS_PATH)


def crop_brain_contour(image):
    """Crops brain area from MRI scan background to focus ROI."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres = cv2.erode(thres, None, iterations=2)
    thres = cv2.dilate(thres, None, iterations=2)
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if cropped.size == 0:
        return image
    return cropped


def compute_gradcam(img_tensor, last_conv_layer_name=None, pred_index=None):
    """Computes Grad-CAM activation heatmap using pre-cached computational graph (0.4s speed)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = app_config.get('target_layer', 'block5_conv4')

    grad_model, used_layer_name = model_manager.get_grad_model(last_conv_layer_name)
    if grad_model is None:
        raise ValueError("Grad-CAM model graph not available.")

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), used_layer_name


def generate_gradcam_overlay(original_img_rgb, heatmap, alpha=None, colormap_name=None):
    """Overlays Grad-CAM heatmap onto the original image using selected colormap."""
    if alpha is None:
        alpha = app_config.get('alpha', 0.45)
    if colormap_name is None:
        colormap_name = app_config.get('colormap', 'JET')

    cv2_colormap = COLORMAP_DICT.get(colormap_name.upper(), cv2.COLORMAP_JET)

    heatmap_resized = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2_colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = (colored_heatmap * alpha + original_img_rgb * (1.0 - alpha)).astype(np.uint8)
    return overlay


def image_to_base64(img_rgb):
    """Encodes RGB image array to PNG base64 string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    model, mtime = model_manager.get_model()
    is_loaded = model is not None
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)) if mtime > 0 else 'N/A'
    return jsonify({
        'status': 'ready' if is_loaded else 'error',
        'model_loaded': is_loaded,
        'weights_filename': 'model_weights.npz',
        'last_weight_reload': mtime_str,
        'target_layer': app_config['target_layer'],
        'colormap': app_config['colormap'],
        'alpha': app_config['alpha'],
        'auto_reload': app_config['auto_reload'],
        'metrics': {
            'train_accuracy': '95.57%',
            'validation_accuracy': '93.23%',
            'test_accuracy': '93.55%'
        }
    })


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global app_config
    if request.method == 'POST':
        data = request.json or {}
        if 'target_layer' in data:
            app_config['target_layer'] = data['target_layer']
        if 'colormap' in data:
            app_config['colormap'] = data['colormap']
        if 'alpha' in data:
            app_config['alpha'] = float(data['alpha'])
        if 'auto_reload' in data:
            app_config['auto_reload'] = bool(data['auto_reload'])

        if data.get('force_reload'):
            model_manager.load_model_if_updated(force=True)

        return jsonify({'success': True, 'config': app_config})
    
    return jsonify({'success': True, 'config': app_config})


@app.route('/api/reports', methods=['GET'])
def get_reports():
    return jsonify({'success': True, 'reports': reports_history})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global report_id_counter, reports_history
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    model, _ = model_manager.get_model()
    if model is None:
        return jsonify({'error': 'Model weights not loaded on server.'}), 500

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_raw is None:
            return jsonify({'error': 'Failed to decode image format.'}), 400

        cropped_bgr = crop_brain_contour(img_raw)
        resized_bgr = cv2.resize(cropped_bgr, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        input_tensor = np.expand_dims(img_rgb / 255.0, axis=0)

        preds = model.predict(input_tensor)[0]
        pred_class_idx = int(np.argmax(preds))
        confidence = float(preds[pred_class_idx])

        class_labels = {0: 'Non-Tumorous', 1: 'Tumorous'}
        predicted_label = class_labels.get(pred_class_idx, 'Unknown')

        # Fast Grad-CAM computation using pre-cached model graph
        heatmap, used_layer = compute_gradcam(input_tensor, pred_index=pred_class_idx)
        gradcam_overlay = generate_gradcam_overlay(img_rgb, heatmap)

        original_b64 = image_to_base64(img_rgb)
        gradcam_b64 = image_to_base64(gradcam_overlay)

        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
        report_entry = {
            'report_id': f"REP-{report_id_counter}",
            'filename': file.filename,
            'timestamp': timestamp_str,
            'prediction': predicted_label,
            'confidence': round(confidence * 100, 2),
            'target_layer': used_layer,
            'original_image': original_b64,
            'gradcam_image': gradcam_b64
        }
        report_id_counter += 1
        reports_history.insert(0, report_entry)

        return jsonify({
            'success': True,
            'report_id': report_entry['report_id'],
            'filename': file.filename,
            'prediction': predicted_label,
            'class_index': pred_class_idx,
            'confidence': round(confidence * 100, 2),
            'raw_scores': [float(p) for p in preds],
            'target_layer': f"Layer: {used_layer} heat distribution",
            'original_image': original_b64,
            'gradcam_image': gradcam_b64
        })

    except Exception as e:
        print(f"[Analyze Error] {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print(f"Starting NeuroScan AI Flask Backend on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
