from pycocoevalcap.cider.cider import Cider

# 创建一个全局的Cider评分器实例
cider_scorer = Cider()

def compute_cider_score(prediction: str, ground_truths: list[str]) -> float:
    if not isinstance(ground_truths, list):
        ground_truths = [str(ground_truths)]
    if not isinstance(prediction, str):
        prediction = str(prediction)

    gts = {0: ground_truths}
    res = {0: [prediction]}

    score, _ = cider_scorer.compute_score(gts, res)
    return score