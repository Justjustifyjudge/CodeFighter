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

# 初始化所有模型（使用lazy loading）
classifier = None
label_encoder = None
classifier_tokenizer = None
generator_model = None
generator_tokenizer = None
reasoning_model = None
reasoning_tokenizer = None
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
    global generator_model, generator_tokenizer
    if generator_model is None or generator_tokenizer is None:
        print("Loading generator model and tokenizer...")
        model_path = "../distilled-model-epoch10"
        generator_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        generator_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        print("Generator model and tokenizer loaded successfully")

def load_reasoning_model():
    global reasoning_model, reasoning_tokenizer
    if reasoning_model is None or reasoning_tokenizer is None:
        print("Loading reasoning model and tokenizer...")
        # 1. 先加载全参数微调模型
        full_sft_model = AutoModelForCausalLM.from_pretrained(
            "../finetune/full_sft_model",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 2. 在全量模型基础上加载LoRA
        reasoning_model = PeftModel.from_pretrained(
            full_sft_model,
            "../finetune/hybrid_finetuned_model"
        )
        
        reasoning_tokenizer = AutoTokenizer.from_pretrained(
            "../finetune/hybrid_finetuned_model",
            trust_remote_code=True
        )
        reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token
        print("Reasoning model and tokenizer loaded successfully")

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

def generate_with_reasoning(problem_text):
    """使用思考模型生成分析"""
    load_reasoning_model()
    
    # 准备输入
    input_text = f"问题: {problem_text}\n分析:"
    inputs = reasoning_tokenizer(input_text, return_tensors="pt").to(reasoning_model.device)
    
    # 生成分析
    outputs = reasoning_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        temperature=0.7
    )
    
    # 解码输出
    reasoning_result = reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取分析部分（去掉原始问题）
    analysis = reasoning_result[len(input_text):].strip()
    
    return analysis

def generate_code_with_model(problem_text, generation_params):
    """使用生成模型生成代码"""
    load_generator()
    
    # 准备输入
    inputs = generator_tokenizer(problem_text, return_tensors="pt").to(generator_model.device)
    
    # 生成输出
    outputs = generator_model.generate(
        **inputs,
        max_length=generation_params.get('max_length', 1024),
        temperature=generation_params.get('temperature', 0.7),
        top_p=generation_params.get('top_p', 0.9),
        do_sample=generation_params.get('do_sample', True)
    )
    
    # 解码输出
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate_code', methods=['POST'])
def generate_code():
    # 获取请求数据并严格检查
    data = request.get_json()
    if not data or 'problem' not in data:
        raise BadRequest("Missing required 'problem' in request data")
    
    input_text = data['problem']
    generation_params = {
        'max_length': data.get('max_length', 1024),
        'temperature': data.get('temperature', 0.7),
        'top_p': data.get('top_p', 0.9),
        'do_sample': data.get('do_sample', True)
    }
    
    # 获取显式指定的难度级别（如果存在）
    explicit_difficulty = data.get('difficulty', None)
    
    # 确定最终使用的难度级别
    if explicit_difficulty:
        # 使用显式指定的难度级别（转换为首字母大写，其余小写的形式）
        difficulty = explicit_difficulty.capitalize()
        # 验证难度级别是否有效
        if difficulty not in ['Easy', 'Medium', 'Hard']:
            return jsonify({
                "status": "error",
                "message": "Invalid difficulty level. Must be one of: Easy, Medium, Hard"
            }), 400
    else:
        # 没有显式指定时，使用分类器预测难度
        try:
            difficulty = predict_difficulty(input_text)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Difficulty prediction failed: {str(e)}"
            }), 500
    
    reasoning_result = None
    final_input = input_text
    
    # 如果是困难问题（无论是显式指定还是分类器预测），先使用思考模型
    if difficulty == "Hard":
        try:
            reasoning_result = generate_with_reasoning(input_text)
            # 将分析结果合并到原始问题中
            final_input = f"{input_text}\n\n分析:\n{reasoning_result}"
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Reasoning generation failed: {str(e)}",
                "difficulty": difficulty,
                "is_explicit_difficulty": explicit_difficulty is not None
            }), 500
    
    # 生成最终代码
    try:
        result = generate_code_with_model(final_input, generation_params)
        
        # 返回结果
        response = {
            "status": "success",
            "difficulty": difficulty,
            "is_explicit_difficulty": explicit_difficulty is not None,
            "result": result,
            "input_length": len(input_text),
            "output_length": len(result)
        }
        
        # 如果是困难问题，返回分析结果
        if reasoning_result:
            response["reasoning"] = reasoning_result
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "difficulty": difficulty,
            "is_explicit_difficulty": explicit_difficulty is not None,
            "reasoning": reasoning_result if reasoning_result else None
        }), 500

# 其他端点保持不变...
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
    app.run(host='0.0.0.0', port=5001, debug=False)