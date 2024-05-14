from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def select_emotion(asr_result):
    semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')
    emotion = semantic_cls(asr_result)
    sorted_data = sorted(zip(emotion['scores'], emotion['labels']))
    # 获取具有最高分数的情感标签
    highest_label = sorted_data[-1][1]
    return highest_label

if __name__=="__main__":
    asr_result ="请问你是谁"
    emotion = select_emotion(asr_result)
    print(emotion)