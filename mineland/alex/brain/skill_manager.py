import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

from ..prompt_template import load_prompt


class skillInfo(BaseModel):
    name: str = Field(description="name")
    description: str = Field(description="description")

class SkillManager:
    def __init__(self,
                 deployment_name = 'gpt-4o-v2',
                 azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
                 api_version = "2024-08-01-preview",
                 max_tokens = 256,
                 temperature = 0,):
        print(f"\n{'='*50}")
        print("Initializing Skill Manager...")
        print(f"Model: {deployment_name}")
        print(f"Endpoint: {azure_endpoint}")
        print(f"API Version: {api_version}")
        
        try:
            model = AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parser = JsonOutputParser(pydantic_object=skillInfo)
            self.chain = model | parser
            print("✅ Successfully initialized Skill Manager")
        except Exception as e:
            print(f"❌ Failed to initialize Skill Manager: {str(e)}")
            raise

        print(f"{'='*50}\n")

    def render_system_message(self):
        prompt = load_prompt("generate_skill_description")
        return SystemMessage(content=prompt)
    
    def render_human_message(self, code_info):
        code = code_info["last_code"]
        human_message = HumanMessage(content=code)
        return human_message
    
    def generate_skill_info(self, code_info):
        print("\nGenerating skill info...")
        try:
            system_message = self.render_system_message()
            human_message = self.render_human_message(code_info)
            message = [system_message, human_message]
            skill_info = self.chain.invoke(message)
            print("✅ Successfully generated skill info")
            print(f"\033[31m****Skill Manager****\n{skill_info}\033[0m")
            return skill_info
        except Exception as e:
            print(f"❌ Failed to generate skill info: {str(e)}")
            raise
