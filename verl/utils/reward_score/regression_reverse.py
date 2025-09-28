import re
from typing import Optional, Dict, Tuple

def extract_answer_text(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract the final answer text inside <answer>...</answer>."""
    # Extract last <answer>...</answer>
    match = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if not match:
        return None
    return match[-1].strip()

def validate_structure(processed_str: str) -> bool:
    """Check whether <think>...</think><answer>...</answer> appear once and in correct order."""
    tags = {
        '<think>': processed_str.count('<think>'),
        '</think>': processed_str.count('</think>'),
        '<answer>': processed_str.count('<answer>'),
        '</answer>': processed_str.count('</answer>'),
    }
    if any(v != 1 for v in tags.values()):
        return False
    return (processed_str.find('<think>') < processed_str.find('</think>') <
            processed_str.find('<answer>') < processed_str.find('</answer>'))

def compute_score(data_source: str,
                 solution_str: str,
                 ground_truth: str,
                 extra_info: Optional[Dict] = None) -> float:
    """Compute reward score based on format and content correctness."""
    # Extract <answer> content and cleaned string
    answer_text = extract_answer_text(solution_str)

    # 1. 格式分数
    format_score = 1 if validate_structure(solution_str) else -1

    # 2. 答案内容分数（直接对比内容）
    gt_answer = float(ground_truth.strip())
    pred_answer = answer_text.strip() if answer_text else ""
    try:
        # 1. 尝试将输出转换成 float
        pred = float(pred_answer)

        pred = max(1.0, min(5.0, pred))

        error = abs(pred - gt_answer)
        if error == 0:
            content_score = 2
        elif error == 1 or error == 4:
            content_score = -2
        else:
            content_score = -1
        acc = pred == gt_answer
    except ValueError:
        pred = 0
        content_score = -2.0
        acc = False
        error = 4
    total_score = format_score + content_score

    # # 可选打印调试
    # print(f"\n[my_reward_fn Debug]")
    # print(f"Format: {'PASS' if format_score > 0 else 'FAIL'} | Content: {'MATCH' if content_score > 0 else 'MISMATCH'}")
    # print(f"GT: {gt_answer}\nPred: {pred_answer}")
    # print(f"Score = {format_score} + {content_score} = {total_score}\n")

    return {
        "score": total_score,
        "acc": acc,
        "mae": error,
        "format_score": format_score,
        "content_score": content_score,
    }