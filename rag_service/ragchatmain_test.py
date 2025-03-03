# from getpass import getpass
# HUGGINGFACEHUB_API_TOKEN = getpass()
HUGGINGFACEHUB_API_TOKEN = ""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage, AIMessage


import os
import time
# from openai import OpenAI
# from dotenv import load_dotenv
import sys
import readline

# load API key from the env
# load_dotenv()

# create OpenAI client and set the API key
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# hf_client = HuggingFaceHub(repo_id="google/flan-t5-base")
llm = HuggingFaceEndpoint(
    # repo_id="microsoft/Phi-3-mini-4k-instruct",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    # repo_id="google/flan-t5-xl",
    task="text-generation",
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.03,
)

def check_farewell_intent(text):
    """
    使用GPT判断用户输入是否包含结束对话的意图
    """
    try:
        
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "你的任务是判断用户的输入是否表达了想要结束对话的意图。如果是，请只回复'true'，如果不是，请只回复'false'。"},
        #         {"role": "user", "content": f"用户说：{text}"}
        #     ],
        #     temperature=0,  # 使用较低的temperature以获得更确定的答案
        #     max_tokens=10   # 只需要简短的回复
        # )
        
        # result = response.choices[0].message.content.strip().lower()
        chat = ChatHuggingFace(llm=llm, verbose=True)

        messages = [
            SystemMessage(content="Your task is to determine whether the user's input expresses an intention to end the conversation. If yes, reply only 'true', if not, reply only 'false'."),
            HumanMessage(content=text),
        ]

        res = chat.invoke(messages)
        return res.lower() == 'true'
    except Exception as e:
        print(f"\n判断意图时发生错误: {str(e)}")
        return False

def create_chat_completion(messages, retries=3, delay=2):
    """
    创建聊天完成，包含重试机制
    """
    for attempt in range(retries):
        try:
            chat = ChatHuggingFace(llm=llm, verbose=True)

            res = chat.invoke(messages)
            # import pdb;pdb.set_trace()
            return res
            # return client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            #     stream=True
            # )
        except Exception as e:
            if attempt == retries - 1:  # 最后一次尝试
                raise e
            print(f"\n发生错误，{delay}秒后重试: {str(e)}")
            time.sleep(delay)

def chat_with_gpt():
    # 初始化消息历史，包含系统角色
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant who will provide accurate, useful answers."}
    # ]
    messages = [
        SystemMessage(content="You are a helpful AI assistant who will provide accurate, useful answers.")
    ]
    
    print("Welcome to rag_service, press 'quit' to leave, 'clear' to reset conversation.")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Bye!")
                break
            
            if user_input.lower() == 'clear':
                messages = [messages[0]]  # 保留系统消息
                print("Content Reset!")
                continue
            
            if not user_input:
                continue
            
            # 检查是否包含告别意图
            # if check_farewell_intent(user_input):
            #     print("Assistant: Thanks for your Question, Bye!")
            #     break
            
            # 将用户输入添加到消息历史
            messages.append(HumanMessage(user_input))
            # messages.append({"role": "user", "content": user_input})
            
            # 创建流式响应
            stream = create_chat_completion(messages)
            # print(stream)
            
            print("Assistant: ", end="", flush=True)
            
            # 收集完整响应
            full_response = ""
            
            # 流式输出回复
            # for chunk in stream:
            #     if chunk.content is not None:
            #         content = chunk.content
            #         print(content, end="", flush=True)
            #         full_response += content
            
            print(stream.content)  # 换行
            
            # 将助手的回复添加到消息历史
            messages.append(AIMessage(stream.content))
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            sys.exit(0)
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    chat_with_gpt()
