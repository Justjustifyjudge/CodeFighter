from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from werkzeug.exceptions import BadRequest
import joblib
from transformers import LongformerTokenizer
import sys
from pathlib import Path
# from peft import LoraConfig, get_peft_model, PeftModel

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

# from flask import Flask, request, jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from werkzeug.exceptions import BadRequest
# import joblib
# from transformers import LongformerTokenizer
# import sys
# from pathlib import Path
from peft import PeftModel
import time
from typing import Tuple

app = Flask(__name__)

# 长参数配置
WINDOW_SIZE = 1024  # 滑动窗口大小
STRIDE = 512       # 滑动步长
MAX_GLOBAL_LENGTH = 4096  # 全局最大长度

# 初始化模型
classifier = None
label_encoder = None
classifier_tokenizer = None
generator_model = None
generator_tokenizer = None
reasoning_model = None
reasoning_tokenizer = None

def load_models():
    """内存优化的模型加载方式"""
    global generator_model, generator_tokenizer
    
    if generator_model is None:
        print("Loading generator model with memory optimization...")
        start_time = time.time()
        
        # 低内存模式加载
        generator_tokenizer = AutoTokenizer.from_pretrained(
            "../distilled-model-epoch3",
            trust_remote_code=True
        )
        
        generator_model = AutoModelForCausalLM.from_pretrained(
            "../distilled-model-epoch3",
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="./offload",
            trust_remote_code=True
        )
        
        print(f"Generator loaded in {time.time()-start_time:.2f}s, Memory: {get_memory_usage()}")

def get_memory_usage() -> str:
    """获取内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"{allocated:.1f}MB/{reserved:.1f}MB"
    return "CPU mode"

def sliding_window_generation(
    prompt: str,
    max_additional_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_retries: int = 3
) -> Tuple[str, dict]:
    """
    增强鲁棒性的滑动窗口生成
    自动调整参数确保始终有输出
    """
    load_models()
    
    # 初始编码
    input_ids = generator_tokenizer.encode(prompt, return_tensors="pt").to(device)
    total_length = min(input_ids.shape[1], MAX_GLOBAL_LENGTH)
    generated = input_ids.clone()
    
    stats = {
        "windows": 0,
        "total_tokens": 0,
        "start_time": time.time(),
        "retries": 0,
        "adjusted_params": {}
    }

    # 参数安全范围
    safe_params = {
        'temperature': (0.3, 1.5),
        'top_p': (0.5, 0.99),
        'max_additional_tokens': (256, 2048)
    }

    def adjust_params(params):
        """智能参数调整"""
        new_params = params.copy()
        # 逐步放宽限制
        new_params['temperature'] = max(min(params['temperature'], safe_params['temperature'][1]), 
                                   safe_params['temperature'][0])
        new_params['top_p'] = max(min(params['top_p'], safe_params['top_p'][1]),
                             safe_params['top_p'][0])
        new_params['max_additional_tokens'] = min(
            params['max_additional_tokens'], 
            safe_params['max_additional_tokens'][1]
        )
        return new_params

    current_params = {
        'temperature': temperature,
        'top_p': top_p,
        'max_additional_tokens': max_additional_tokens
    }

    for attempt in range(max_retries + 1):
        try:
            with torch.no_grad():
                while total_length < MAX_GLOBAL_LENGTH:
                    window_start = max(0, generated.shape[1] - WINDOW_SIZE)
                    window_ids = generated[:, window_start:]
                    
                    generation_config = {
                        "input_ids": window_ids,
                        "max_length": window_ids.shape[1] + current_params['max_additional_tokens'],
                        "temperature": current_params['temperature'],
                        "top_p": current_params['top_p'],
                        "do_sample": True,
                        "pad_token_id": generator_tokenizer.eos_token_id,
                        "repetition_penalty": 1.1,
                        "typical_p": 0.9,
                        "num_return_sequences": 1
                    }

                    outputs = generator_model.generate(**generation_config)
                    new_tokens = outputs[:, window_ids.shape[1]:]
                    
                    # 检查有效输出
                    if new_tokens.shape[1] == 0:
                        raise ValueError("Empty generation")
                        
                    generated = torch.cat([generated, new_tokens], dim=-1)
                    stats["windows"] += 1
                    stats["total_tokens"] += new_tokens.shape[1]
                    total_length = generated.shape[1]
                    
                    # 遇到结束符或达到长度限制则停止
                    if new_tokens[0, -1] == generator_tokenizer.eos_token_id:
                        break

            # 生成成功则记录最终参数
            stats['adjusted_params'] = current_params
            break

        except Exception as e:
            if attempt == max_retries:
                # 最后一次尝试仍失败，返回已生成内容
                stats['error'] = str(e)
                break
                
            # 调整参数再次尝试
            current_params = adjust_params(current_params)
            stats['retries'] += 1
            stats['adjusted_params'] = current_params
            print(f"Retry {attempt+1} with adjusted params: {current_params}")

    stats.update({
        "time_elapsed": time.time() - stats["start_time"],
        "memory_usage": get_memory_usage(),
        "final_length": total_length,
        "completed": 'error' not in stats
    })
    
    return generator_tokenizer.decode(generated[0], skip_special_tokens=True), stats

@app.route('/generate_code', methods=['POST'])
def generate_code():
    data = request.get_json()
    if not data or 'problem' not in data:
        raise BadRequest("Missing required 'problem' in request data")
    
    try:
        # 1. 处理难度预测
        difficulty = (predict_difficulty(data['problem']) 
                    if not data.get('difficulty') 
                    else data['difficulty'].capitalize())
        
        # 2. 思考模型处理
        final_input = data['problem']
        reasoning_result = None
        if difficulty == "Hard":
            try:
                reasoning_result = generate_with_reasoning(data['problem'])
                final_input = f"{data['problem']}\n\n分析:\n{reasoning_result}"
            except Exception as e:
                print(f"Reasoning model failed, using original input: {str(e)}")

        # 3. 带自动修复的生成
        result, stats = sliding_window_generation(
            prompt=final_input,
            max_additional_tokens=int(data.get('max_length', 1024)),
            temperature=float(data.get('temperature', 0.7)),
            top_p=float(data.get('top_p', 0.9))
        )
        
        response = {
            "status": "success" if stats.get("completed") else "partial",
            "difficulty": difficulty,
            "result": result,
            "stats": {
                "windows": stats["windows"],
                "tokens": stats["total_tokens"],
                "time_elapsed": stats["time_elapsed"],
                "adjusted_params": stats.get("adjusted_params", {})
            }
        }
        
        if reasoning_result:
            response["reasoning"] = reasoning_result
            
        return jsonify(response)
    
    except Exception as e:
        # 终极fallback方案
        try:
            minimal_output = generator_tokenizer.decode(
                generator_tokenizer.encode(data['problem'])[:100],
                skip_special_tokens=True
            )
            return jsonify({
                "status": "fallback",
                "result": minimal_output,
                "message": f"System recovered with minimal output: {str(e)}"
            })
        except:
            return jsonify({
                "status": "error",
                "message": "Complete generation failure",
                "suggestion": "Please check input format and try again"
            }), 500

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
        model_path = "../distilled-model-epoch3"
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
    app.run(host='0.0.0.0', port=5000, debug=False)