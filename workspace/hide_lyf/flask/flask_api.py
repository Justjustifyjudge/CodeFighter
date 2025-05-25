from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from werkzeug.exceptions import BadRequest
import joblib
from transformers import LongformerTokenizer
import sys
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel

# 将上一级目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 然后可以正常导入
from classification.model import DifficultyClassifier
app = Flask(__name__)

# 初始化分类器和大模型（使用lazy loading）
classifier = None
label_encoder = None
classifier_tokenizer = None
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_classifier():
    global classifier, label_encoder, classifier_tokenizer
    if classifier is None or label_encoder is None or classifier_tokenizer is None:
        print("Loading classifier components...")
        # 分类器相关组件
        classifier_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        label_encoder = joblib.load('../classification/label_encoder.pkl')
        
        # 加载模型 (需提前定义好DifficultyClassifier类)
        classifier = DifficultyClassifier(len(label_encoder.classes_)).to(device)
        classifier.load_state_dict(torch.load('../classification/best_model.bin', map_location=device))
        classifier.eval()
        print("Classifier components loaded successfully")

def load_generator():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading generator model and tokenizer...")
        model_path = "../distilled-model-epoch10"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        print("Generator model and tokenizer loaded successfully")

def predict_difficulty(text, max_length=1024):
    """预测文本难度"""
    # 确保分类器已加载
    load_classifier()
    
    # 文本编码
    encoded = classifier_tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 预测
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = classifier(input_ids, attention_mask)
        pred_idx = torch.argmax(outputs).item()
    
    return label_encoder.inverse_transform([pred_idx])[0]

@app.route('/generate_code', methods=['POST'])
def generate_code():
    # 获取请求数据
    data = request.get_json()
    if not data or 'problem' not in data:
        raise BadRequest("Missing 'problem' in request data")
    
    input_text = data['problem']
    
    # 先预测难度
    try:
        difficulty = predict_difficulty(input_text)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Difficulty prediction failed: {str(e)}"
        }), 500
    
    # 然后生成代码
    try:
        # 确保生成模型已加载
        load_generator()
        
        # 可选参数
        max_length = data.get('max_length', 1024)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        do_sample = data.get('do_sample', True)
        
        # 准备输入
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # 生成输出
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        # 解码输出
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 返回结果（包含难度信息）
        return jsonify({
            "status": "success",
            "difficulty": difficulty,
            "result": result,
            "input_length": len(input_text),
            "output_length": len(result)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "difficulty": difficulty  # 即使生成失败也返回已预测的难度
        }), 500

@app.route('/predict_difficulty', methods=['POST'])
def predict_difficulty_endpoint():
    """单独预测难度的接口"""
    data = request.get_json()
    if not data or 'problem' not in data:
        raise BadRequest("Missing 'problem' in request data")
    
    try:
        difficulty = predict_difficulty(data['problem'])
        return jsonify({
            "status": "success",
            "difficulty": difficulty
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)