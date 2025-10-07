import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import re,ast
import numpy as np 
from navsim.common.waymo_utils import interpolate_trajectory, get_rater_feedback_score
from string import Template

# ---------- 2. 字段提取函数 ---------- #

def load_json(path) -> List[Dict[str, Any]]:
    """加载标准 JSON（整体是列表的结构）"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_from_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从列表-JSON 中抽取关键信息"""
    result = []
    for smp in samples:
        meta = smp.get("meta_info", {})
        # 找到 assistant 生成的预测文本（若有）
        pred_txt = next(
            (m["content"] for m in smp.get("messages", [])
             if m.get("role") == "assistant" and "Predicted" in m.get("content", "")),
            None
        )
        result.append(
            {
                "id": smp.get("id"),
                "initial_speed": meta.get("initial_speed"),
                "trajectory": meta.get("trajectory"),      # 轨迹点列表
                "rfs_scores": meta.get("rfs_scores"),
                "rfs_trajs": meta.get("rfs_trajs"),
                "rfs_len": meta.get("rfs_len"),
                "predicted_waypoints": pred_txt,
            }
        )
    return result

def load_annotation(path):
    """
    Load annotations from a JSONL file.

    Args:
        path (str): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents an annotation.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:                       # 跳过空行
                records.append(json.loads(line))
    result = []
    for rec in records:
        pred_str = rec.get("predict")
        pred_str = pred_str.strip()

    # 2) 用正则一次性删除开头 ```json（或简单 ```）和结尾 ```
        pred_str = re.sub(r'^```(?:json)?\s*', '', pred_str)   # 去起始行
        pred_str = re.sub(r'\s*```$', '', pred_str)            # 去结束行
        pred_str = json.loads(pred_str)
        result.append(pred_str)
                
    return result
    
def extract_trajs(pred_text):

    pattern = r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, pred_text)

    if not matches:
        raise ValueError("在文本中没有找到任何坐标！")

    # 转成浮点，然后堆叠为 (N, 2) 形状
    coords = np.array([[float(x), float(y)] for x, y in matches], dtype=float)

    return coords


def extract_state(block: str):
    # intent
    intent = re.search(r'high-level intent:\s*([A-Z_]+)', block, re.I).group(1)

    # acceleration and velocity
    acc_str, vel_str = re.search(
        r'acceleration and velocity:\s*\[([^\]]+)\],\s*\[([^\]]+)\]',
        block, re.I
    ).groups()

    acc = [float(v) for v in acc_str.split(',')]
    vel = [float(v) for v in vel_str.split(',')]

    return vel, acc,intent



system_prompt = "You are an expert driver."

PROMPT_TEMPLATE = Template(
    "Input:\n"
    "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep\n"
    "1) Front view: <image>\n"
    "2) Front-right view: <image>\n"
    "3) Front-left view: <image>\n"
    "- Current high-level intent: $driving_cmd\n"
    "- Current acceleration and velocity: [$acc_x,$acc_y], [$vel_x,$vel_y]\n"
    "Task 1: Critical Objects and Conditions Detection\n"
    "Decide whether at least one critical instance of each class could influence the ego-vehicle’s future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego’s future path. Output \"yes\" or \"no\" for every class (no omissions).\n"
    "Object classes to audit:\n"
    "- nearby_vehicle\n"
    "- pedestrian\n"
    "- cyclist\n"
    "- construction\n"
    "- traffic_element\n"
    "- weather_condition\n"
    "- road_hazard\n"
    "- emergency_vehicle\n"
    "- animal\n"
    "- special_vehicle\n"
    "- conflicting_vehicle\n"
    "- door_opening_vehicle\n"
    "Output format (strict JSON, no extra keys, no commentary):\n"
    "{\n"
    "  \"critical_objects\": {\n"
    "    \"nearby_vehicle\": \"yes|no\",\n"
    "    \"pedestrian\": \"yes|no\",\n"
    "    \"cyclist\": \"yes|no\",\n"
    "    \"construction\": \"yes|no\",\n"
    "    \"traffic_element\": \"yes|no\",\n"
    "    \"weather_condition\": \"yes|no\",\n"
    "    \"road_hazard\": \"yes|no\",\n"
    "    \"emergency_vehicle\": \"yes|no\",\n"
    "    \"animal\": \"yes|no\",\n"
    "    \"special_vehicle\": \"yes|no\",\n"
    "    \"conflicting_vehicle\": \"yes|no\",\n"
    "    \"door_opening_vehicle\": \"yes|no\"\n"
    "  },\n"
    "Task 2: Natural Language Explanation\n"
    "Compose a concise natural-language description of the optimal future 5-second trajectory for the ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute this trajectory.\n"
    "- Mention only the classes you marked \"yes\" in the previous task.\n"
    "- Describe how each of those critical objects or conditions influences the optimal trajectory.\n"
    "- Do not invent objects or conditions not present in the input.\n"
    "Output format (strict JSON, no extra keys, no commentary):\n"
    "{\n"
    "  \"explanation\": \"100-word description that references only the classes marked \'yes\'\"\n"
    "}\n"
    "Task 3: Meta-Behaviour Selection\n"
    "Assign exactly one category from each list. Choose the label that best summarises the overall behaviour of the optimal future trajectory:.\n"
    "- speed ∈ { keep, accelerate, decelerate }\n"
    "- command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }\n"
    "- If none fits, use 'other', but do this sparingly.\n"
    "Output format (strict JSON, no extra keys, no commentary):\n"
    "{\n"
    "\"meta_behaviour\": {\n"
    "\"speed\": \"keep | accelerate | decelerate | other\",\n"
    "\"command\": \"straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other\"}\n"
    "}\n"
    "Task 4: Future Trajectory Prediction\n"
    "Given the input, critical objects/conditions, natural language explanation, and meta-behaviour, predict the optimal 5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
    "Output format (raw text, not markdown or LaTeX):\n"
    "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]"
)

# -------------------------------------------------------------------
# 3. Helper to build the ready-to-send prompt from a raw description
# -------------------------------------------------------------------
def make_prompt(raw_description: str) -> str:
    """
    Convert a raw scenario description into the final prompt expected by
    the large-language model.

    Returns
    -------
    str
        Prompt string with variables interpolated.
    """
    vel, acc, cmd = extract_state(raw_description)
    return PROMPT_TEMPLATE.substitute(
        vel_x=vel[0],
        vel_y=vel[1],
        acc_x=acc[0],
        acc_y=acc[1],
        driving_cmd=cmd,
    )
# ---------- 3. 演示入口 ---------- #
if __name__ == "__main__":


    samples_path = "/home/fenglan/DiffusionDrive/navsim/planning/script/val_merged.json"
    #jsonl_path    = "/home/fenglan/DiffusionDrive/navsim/planning/script/qwen_waymo_annotator.jsonl"

    samples_raw = load_json(samples_path)
    #jsonl_info   = load_annotation(jsonl_path)

    generated_qas = []
    for i in tqdm(range(len(samples_raw))):
        generated_qa = {}
        generated_qa['images'] = samples_raw[i]['images']

       # driving_cmd, past_txt = extract_values(samples_raw[i]['messages'][0]['content'])
        prompt = make_prompt(samples_raw[i]['messages'][0]['content'])

        # answer_dict = jsonl_info[i]
        # future_txt = samples_raw[i]['messages'][1]['content']
        # full_answer = json.dumps(answer_dict) + '\n' + future_txt

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": "",
            },
        ]
        generated_qa['messages'] = messages
        generated_qa['system'] = system_prompt
        generated_qas.append(generated_qa)

    with open("/home/fenglan/DiffusionDrive/navsim/planning/script/waymo_val_annotated.json", "w") as f:
        json.dump(generated_qas, f, indent=2)
