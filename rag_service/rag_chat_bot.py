HUGGINGFACEHUB_API_TOKEN = ""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
import json
import time
import sys
from typing import List, Dict, Any, Optional
import readline

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from ragdatabase import RAGDatabase

class ChatCompletionHandler:
    """处理与OpenAI API交互的类"""
    def __init__(self):
        self.llm_client = HuggingFaceEndpoint(
            # repo_id="microsoft/Phi-3-mini-4k-instruct",
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            # repo_id="meta-llama/Llama-3.1-8B-Instruct",
            # repo_id="google/flan-t5-xl",
            task="text-generation",
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.03,
        )

    def create_completion(self, messages: List, retries: int = 3, delay: int = 2):
        """创建聊天完成，包含重试机制"""
        for attempt in range(retries):
            try:
                chat = ChatHuggingFace(llm=self.llm_client, verbose=True)
                return chat.invoke(messages).content
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                print(f"\n发生错误，{delay}秒后重试: {str(e)}")
                time.sleep(delay)

    def check_farewell_intent(self, text: str) -> bool:
        """检查是否有结束对话的意图"""
        try:
            chat = ChatHuggingFace(llm=self.llm_client, verbose=True)

            messages = [
                SystemMessage(content="Your task is to determine whether the user wants to end the conversation. If yes, reply one word true, if not, reply one word false."),
                HumanMessage(content=text),
            ]
        
            res = chat.invoke(messages)
            import pdb;pdb.set_trace()
            return res.content.lower() == 'true'
        except Exception as e:
            print(f"\n判断意图时发生错误: {str(e)}")
            return False

class ConversationManager:
    """管理对话历史的类"""
    def __init__(self):
        self.messages = [SystemMessage("You are a knowledgeable AI assistant who answers questions based on the contextual information provided. If context information is insufficient, the user is explicitly informed.")]

    def add_message(self, role: str, content: str):
        """添加新消息"""
        if role == 'system':
            self.messages.append(SystemMessage(content))
        elif role == 'human':
            self.messages.append(HumanMessage(content))
        else:
            self.messages.append(AIMessage(content))

    def clear_history(self):
        """清除对话历史"""
        system_message = self.messages[0]
        self.messages = [system_message]

    def get_messages_dict(self) -> List[Dict[str, str]]:
        """获取消息字典列表"""
        return [msg.to_dict() for msg in self.messages]

class RAGChatBot:
    """RAG增强的聊天机器人主类"""
    def __init__(self):
        self.completion_handler = ChatCompletionHandler()
        self.rag_database = RAGDatabase()
        self.conversation = ConversationManager()

    def _get_relevant_context(self, query: str, k: int = 3) -> str:
        """获取相关上下文"""
        try:
            results = self.rag_database.similarity_search(query, k)
            return '\n'.join(t.page_content for t in results)
        except Exception as e:
            print(f"获取上下文时发生错误: {str(e)}")
            return ""

    def _generate_prompt_with_context(self, query: str, context: str) -> str:
        """生成包含上下文的prompt"""
        return f"""Answer user questions based on the following reference information. If the reference information is not sufficient to answer the question, state that it is not available or that more information is needed.

Reference information:
{context}

User's question: {query}

Answer: """

    def _process_user_input(self, user_input: str) -> Optional[str]:
        """处理用户输入，返回None表示继续对话，返回字符串表示特殊指令的响应"""
        if user_input.lower() == 'quit':
            return "Bye!"
        
        if user_input.lower() == 'clear':
            self.conversation.clear_history()
            return "Reset clear ... "
            
        if not user_input:
            return None
            
        if self.completion_handler.check_farewell_intent(user_input):
            return "Thank you, Bye!"
            
        return None

    def _handle_chat_response(self, stream) -> str:
        """处理聊天响应流"""
        print("ChatGPT: ", end="", flush=True)
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()
        return full_response

    def chat(self):
        """主聊天循环"""
        print("Welcome to RAG chat! quit to leave, clear to reset ...")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # 处理特殊指令
                result = self._process_user_input(user_input)
                if result is not None:
                    print("Assistant: ", result)
                    if result in ["Bye!", "Thank you, Bye!"]:
                        break
                    continue

                # 获取上下文并生成prompt
                context = self._get_relevant_context(user_input)
                prompt_with_context = self._generate_prompt_with_context(user_input, context)
                
                # 添加用户消息并获取回复
                self.conversation.add_message("human", prompt_with_context)
                stream = self.completion_handler.create_completion(self.conversation.messages)
                
                # 处理回复
                # full_response = self._handle_chat_response(stream)
                print(stream)
                full_response = stream
                self.conversation.add_message("assistant", full_response)

            except KeyboardInterrupt:
                print("\n程序被用户中断")
                sys.exit(0)
            except Exception as e:
                print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    rag_bot = RAGChatBot()
    rag_bot.chat()
