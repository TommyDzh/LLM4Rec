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
    try:
        gt_answer = float(ground_truth.strip())
        pred_answer = float(answer_text.strip())
    

        content_score = 2 if pred_answer == gt_answer else -2
        error = abs(pred_answer - gt_answer)
    except ValueError:
        content_score = -2.0
        error = 0

    total_score = format_score + content_score

    # # 可选打印调试
    # print(f"\n[my_reward_fn Debug]")
    # print(f"Format: {'PASS' if format_score > 0 else 'FAIL'} | Content: {'MATCH' if content_score > 0 else 'MISMATCH'}")
    # print(f"GT: {gt_answer}\nPred: {pred_answer}")
    # print(f"Score = {format_score} + {content_score} = {total_score}\n")

    return {
        "score": total_score,
        "mae": error,
        "format_score": format_score,
        "content_score": content_score,
    }
