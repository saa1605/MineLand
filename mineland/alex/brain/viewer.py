import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

from ..prompt_template import load_prompt


class VisionInfo(BaseModel):
    image_summary: str = Field(description="image-summary")

class Viewer():
    def __init__(self, 
                 deployment_name = 'gpt-4o-v2',
                 azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
                 api_version = "2024-08-01-preview",
                 max_tokens = 256,
                 temperature = 0,):
        print(f"\n{'='*50}")
        print("Initializing Viewer...")
        print(f"Model: {deployment_name}")
        print(f"Endpoint: {azure_endpoint}")
        print(f"API Version: {api_version}")
        
        try:
            vlm = AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parser = JsonOutputParser(pydantic_object=VisionInfo)
            self.chain = vlm | parser
            print("✅ Successfully initialized Viewer")
        except Exception as e:
            print(f"❌ Failed to initialize Viewer: {str(e)}")
            raise

        print(f"{'='*50}\n")
    
    def render_system_message(self):
        prompt = load_prompt("vision_summary")
        return SystemMessage(content=prompt)
    
    def render_human_message(self, obs):
        observation = []
        observation.append({"type": "text", "text": str(obs)})
        try:
            image_base64 = obs["rgb_base64"]
            if image_base64 != "":
                observation.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        })
        except:
            print("No image in observation")
            raise Exception("No image in observation")
        
        human_message = HumanMessage(content=observation)
        return human_message
    
    def summary(self, obs):
        print("\nGenerating vision summary...")
        try:
            system_message = self.render_system_message()
            human_message = self.render_human_message(obs)
            message = [system_message, human_message]
            vision_summary = self.chain.invoke(message)
            print("✅ Successfully generated vision summary")
            print(f"\033[31m****Vision Agent****\n{vision_summary}\033[0m")
            return vision_summary
        except Exception as e:
            print(f"❌ Failed to generate vision summary: {str(e)}")
            if "No image in observation" in str(e):
                print("Warning: No image found in observation data")
            raise
