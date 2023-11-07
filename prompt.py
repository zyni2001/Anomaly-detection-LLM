import openai

import json

with open(
    "/Users/tanglu/Desktop/Anomaly-detection-LLM-main/Data/PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json",
    "r",
) as f:
    data = json.load(f)
import random


def generate_prompt(
    reviews
):
    """
        test_case:

    """
    detect_Prompt = """Please retrieve the potential outlier reviews(with rating labels) from Yelp. Rate their outlier score on a scale from 0 to 1, and explain why"""
    prompt = detect_Prompt
    for i in reviews["0"][:3]:
        rating_score =i[3]
        review_label = i[2]
        user_label = i[1]
        user_id = i[0]
        review = i[4]
        prompt+=f"rating_score: {rating_score},review:{review}"
    return prompt
    # chatGPT_generate(prompt)
    # GTAN test集合去做，但是除了test集合之外我们都可以用来做cot的exmaple

CoT_prompt = """
reviews ID: {review2}
confidence for outlier: MEDIUM
reasons: The review mentions having a picnic dinner and getting frisky on the 17th floor roof, which may not be a typical experience for a hotel stay. Additionally, the mention of waking up to an army of ants going in on picnic leftovers may indicate a cleanliness issue, which could be a negative aspect of the hotel.

reviews ID: {review3}
confidence for outlier: LOW
reasons: The review expresses a positive sentiment towards the hotel, mentioning that it reminds the reviewer of places from movies and describing it as scuzzy and sleazy, which the reviewer loves. While the review may not align with typical positive reviews, it does not contain any extreme or contradictory statements.

reviews ID: {review4}
confidence for outlier: HIGH
reasons: The review only contains a link to a YouTube video and does not provide any information or opinion about the hotel. This suggests that the review may be spam or irrelevant to the hotel's actual quality.


# """
def parse_answer(
    prompt,
    ):
    """"
    把大括号里的内容正则匹配
    """
    pass

def llama_generate(prompt):
    # response = openai.Completion.create(engine="", prompt=prompt) 
    pass
def generate_mapper_prompt(
    reviews
):
    """
    
CoT_prompt = 
reviews ID: {review2}
confidence for outlier: MEDIUM
reasons: The review mentions having a picnic dinner and getting frisky on the 17th floor roof, which may not be a typical experience for a hotel stay. Additionally, the mention of waking up to an army of ants going in on picnic leftovers may indicate a cleanliness issue, which could be a negative aspect of the hotel.

reviews ID: {review3}
confidence for outlier: LOW
reasons: The review expresses a positive sentiment towards the hotel, mentioning that it reminds the reviewer of places from movies and describing it as scuzzy and sleazy, which the reviewer loves. While the review may not align with typical positive reviews, it does not contain any extreme or contradictory statements.

reviews ID: {review4}
confidence for outlier: HIGH
reasons: The review only contains a link to a YouTube video and does not provide any information or opinion about the hotel. This suggests that the review may be spam or irrelevant to the hotel's actual quality.

"""
    detect_Prompt = """
    Give you a series of reviews on Yelp and its rating score, the reviews is all from one target(hotel, food), please detect all the reviews outlier or not for each review, Give your answer in the form below:
        reviews ID:<give the reviews ID for rating the confidence as HIGH and MEDIUM>
        confidence for outlier: <one of: LOW, HIGH,MEDIUM>;
        reasons:<text for why it is outlier>;"""
    prompt = detect_Prompt
    num=0
    ground_truth = []
    for i in reviews["0"][0:10]:
        rating_score =i[3]
        review_label = i[2]
        user_label = i[1]
        user_id = i[0]
        review = i[4]
        num+=1
        ground_truth.append({str(num):review_label})
        prompt+=f"""review{num}: {review}rating score on Yelp:{rating_score}\n"""
    return prompt,ground_truth



def chatGPT_generate(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    flag = True
    i = 0
    # 请教我一个数学题目
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,  # 35-turbo
        messages=messages,
        temperature=0, #temperature是0的时候，给定固定的input，他的每次输出都是固定的，当你调高温度的时候，每次输出都会不一样
        max_tokens=300, #input的token限制是多少
        # top_p=0.9, # 当你让你模型做inference的时候，他选择前多少个作为贪心策略
        frequency_penalty=0,
    )
    # self consistency 投票法
    result = response["choices"][0]["message"]["content"]
    return result

# only prompt and answer via chatGPT generate func
# data = random.sample(data, 30)
prompt,ground_truth = generate_mapper_prompt(data)
result = chatGPT_generate(prompt)
print("prompt:",prompt)
print("result",result)
print("ground truth:",ground_truth)