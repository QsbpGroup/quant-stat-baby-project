import os
import re
import time
import openai
import random
import requests
import pandas as pd
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")


def crawler(stock_code='CSI000941', save=False):
    """
    Crawler for xueqiu.com

    Input:
    ------
    url: url for xueqiu.com
    save: save data or not

    Output:
    -------
    return: comments(list)
    """

    url_type = {
        'SH': 13,
        'SZ': 11,
        'CS': 26
    }

    url_response = []
    for page in tqdm(range(1, 101), desc='Crawling', leave=False):
        url = 'https://xueqiu.com/query/v1/symbol/search/status.json?count=10&comment=0&symbol=' +\
            stock_code+'&hl=0&source=all&sort=alpha&page=' + \
            str(page)+'&q=&type='+str(url_type.get(stock_code[:2], 26))
        flag = 0
        while flag == 0:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                       "Cookie": "aliyungf_tc=AQAAADriOUCilQoAxZ5btPQfYv7152ox; acw_tc=2760824915856669537353368e2ea5d4c1b87e45dadece330ae07e755b96f1; xq_a_token=2ee68b782d6ac072e2a24d81406dd950aacaebe3; xqat=2ee68b782d6ac072e2a24d81406dd950aacaebe3; xq_r_token=f9a2c4e43ce1340d624c8b28e3634941c48f1052; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOi0xLCJpc3MiOiJ1YyIsImV4cCI6MTU4NzUyMjY2MSwiY3RtIjoxNTg1NjY2OTA4NDgwLCJjaWQiOiJkOWQwbjRBWnVwIn0.YCQ_yUlzhRvTiUgz1BWWDFrsmlxSgsbaaKs0cxsdxnOaMhIjF0qUX-5WNeqfRXe15I5cPHiFf-5AzeRZgjy0_bSId2-jycpDWuSIseOY07nHM306A8Y1vSJJx4Q9gFnWx4ETpbdu1VXyMYKpwVIKfmSb5sbGZYyHDJPQQuNTfIAtPBiIeHWPDRB-wtf0qa5FNSMK3LKHRZooXjUgh-IAFtQihUIr9D81tligmjNYREntMY1gLg5Kq6GjgivfF9CFc11sJ11fZxnSw9e8J_Lmx8XXxhwHv-j4-ANUSIuglM4cT6yCsWa3pGAVMN18r2cV72JNkk343I05DevQkbX8_A; u=481585666954081; Hm_lvt_1db88642e346389874251b5a1eded6e3=1585666971; device_id=24700f9f1986800ab4fcc880530dd0ed; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1585667033"}
            response = requests.get(url,  headers=headers)
            temp = response.json()
            if 'code' in temp:  # 访问失败
                print(temp, headers)
                time.sleep(0.5)
            else:  # 访问成功，不用再访问，跳出循环
                flag = 1
            url_response.extend(temp['list'])
    data_list = _get_comment(url_response)

    if save:
        data_df = pd.DataFrame(data_list, columns=['text', 'comment_time', 'title', 'like_count', 'reply_count', 'favorite_count',
                                                   'view_count', 'retweet_count', 'is_hot', 'is_answer', 'is_bonus', 'is_reward', 'reward_count', 'screen_name'])
        data_df.to_csv("./comment_data.csv", encoding="utf_8_sig",
                       index=False, header=True)
    return data_df


def _get_comment(data):
    """
    Get comments from xueqiu.com, includingtext, comment_time, title, like_count, reply_count, favorite_count, view_count, retweet_count, is_hot, is_answer, is_bonus, is_reward, reward_count, screen_name

    Input:
    ------
    data: data from xueqiu.com

    Output:
    -------
    return: comments
    """
    data_list = []
    pinglun_len = len(data)
    i = 0
    print('Number of comments:', pinglun_len)

    while i < pinglun_len:
        temp_data = data[i]
        des = '>' + temp_data['description'] + '<'
        pre = re.compile('>(.*?)<')
        text = ''.join(pre.findall(des))
        # convert timestamp into real time
        timeArray = time.localtime(temp_data['created_at'] / 1000 + 11*3600)
        comment_time = time.strftime("%Y-%m-%d %H:%M", timeArray)
        title = temp_data['title']
        like_count = temp_data['like_count']
        reply_count = temp_data['reply_count']
        favorite_count = temp_data['fav_count']
        view_count = temp_data['view_count']
        retweet_count = temp_data['retweet_count']
        is_hot = temp_data['hot']
        is_answer = temp_data['is_answer']
        is_bonus = temp_data['is_bonus']
        is_reward = temp_data['is_reward']
        reward_count = temp_data['reward_count']
        screen_name = temp_data['user']['screen_name']
        data_list.append([text, comment_time, title, like_count, reply_count, favorite_count, view_count,
                          retweet_count, is_hot, is_answer, is_bonus, is_reward, reward_count, screen_name])
        i += 1
    return data_list


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_sentiment(comment):
    """
    Get sentiment from comment

    Input:
    ------
    prompt: prompt for openai

    Output:
    -------
    return: sentiment(one of Bullish, Bearish, Neutral)
    """
    system_prompt = """ 
    You are now acting as an experienced stock market manager. The user will provide you with a snippet of discussion from a Chinese stock market forum regarding a specific stock or sector. Your task is to evaluate the sentiment expressed by the individual who posted the text about the stock or sector in question. You are good at analysing sentiment from the Chinese stock market forum.
    Output Format: reply only one of the following: Bullish, Bearish, or Neutral. Prioritize determining whether the sentiment is Bullish or Bearish; only use "Neutral" if the sentiment is genuinely ambiguous or unclear.
    """
    times = 0
    while True:
        times += 1
        response = chat_completions_with_backoff(
            model="gpt-4",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comment},
            ]
        )
        sentiment = response['choices'][0]['message']['content']
        # count words
        if len(sentiment.split()) == 1:
            break
        else:
            if times > 2:
                sentiment = 'Neutral'
                break
            continue
    return sentiment


if __name__ == '__main__':
    file_name = 'comment_data.csv'
    if os.path.exists(os.path.join(os.getcwd(), file_name)):
        comment_df = pd.read_csv(file_name)
        if len(comment_df) < 900:
            comment_df = crawler(save=True)
    else:
        comment_df = crawler(save=True)
    sentiments = []
    for i in tqdm(range(len(comment_df))):
        # make sure not to exceed the rate limit of API
        if (i+1) % 20 == 0:
            time.sleep(10)
        comment = comment_df['text'][i]
        sentiment = get_sentiment(comment)
        sentiments.append(sentiment)
    comment_df['sentiment'] = sentiments
    comment_df.to_csv("./comment_data_with_sentiment.csv",
                      encoding="utf_8_sig", index=False, header=True)
