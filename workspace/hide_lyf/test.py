from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定保存路径
model_path = "distilled-model-epoch10"  # 根据实际保存的epoch数修改

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 准备输入
input_text = '给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。 你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。 你可以按任意顺序返回答案。   示例 1： 输入：nums = [2,7,11,15], target = 9 输出：[0,1] 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。 示例 2： 输入：nums = [3,2,4], target = 6 输出：[1,2] 示例 3： 输入：nums = [3,3], target = 6 输出：[0,1]   提示： 2 <= nums.length <= 104 -109 <= nums[i] <= 109 -109 <= target <= 109 只会存在一个有效答案   进阶：你可以想出一个时间复杂度小于 O(n2) 的算法吗？请为代码加上注释解释'
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 生成输出
outputs = model.generate(
    **inputs,
    max_length=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)