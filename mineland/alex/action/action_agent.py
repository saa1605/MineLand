import os

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

from ... import Action
from ..prompt_template import load_prompt


class ActionInfo(BaseModel):
    Explain: str = Field(description="Explain")
    Plan: str = Field(description="Plan")
    Code: str = Field(description="Code")

class ActionAgent():
    def __init__(self,
                 deployment_name = 'gpt-4o-v2',
                 azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
                 api_version = "2024-08-01-preview",
                 max_tokens = 1024,
                 temperature = 0,
                 save_path = "./save",):
        print(f"\n{'='*50}")
        print("Initializing Action Agent...")
        print(f"Model: {deployment_name}")
        print(f"Endpoint: {azure_endpoint}")
        print(f"API Version: {api_version}")
        
        self.save_path = save_path
        
        try:
            model = AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_tokens=max_tokens,
                temperature=temperature
            )
            parser = JsonOutputParser(pydantic_object=ActionInfo)
            self.chain = model | parser
            print("✅ Successfully initialized Action Agent")
        except Exception as e:
            print(f"❌ Failed to initialize Action Agent: {str(e)}")
            raise

        print(f"{'='*50}\n")

    def render_system_message(self):
        system_template = load_prompt("high_level_action_template")
        #FIXME: fix program loading
        programs = load_prompt("programs")
        code_example = load_prompt("code_example")
        response_format = load_prompt("high_level_action_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs,
            code_example=code_example,
            response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_human_message(self, obs, short_term_plan, code_info=None, critic_info=None):
        content = []
        text = ""
        text += f"short-term plan: {short_term_plan}\n"
        text += f"observation: {str(obs)}\n"
        if code_info is not None:
            text += f"code info: {code_info}\n"
        if critic_info is not None:
            text += f"critic info: {critic_info}\n"
        content.append({"type": "text", "text": text,})
        try:
            image_base64 = obs["rgb_base64"]
            if image_base64 != "":
                content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        })
        except:
            pass
        human_message = HumanMessage(content=content)
        return human_message
    
    def execute(self, obs, short_term_plan, max_tries = 3, verbose = False):
        print("\nExecuting action plan...")
        try:
            system_message = self.render_system_message()
            human_message = self.render_human_message(obs, short_term_plan)
            message = [system_message, human_message]

            try:
                response = self.chain.invoke(message)
                if verbose:
                    print("✅ Successfully generated action")
                    print(f"\033[31m****Action Agent****\n{response}\033[0m")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write(f"****Action Agent****\n{response}\n")
                return Action(type=Action.NEW, code=response["Code"])
            except Exception as e:
                print(f"⚠️ Failed to generate action, retrying... ({max_tries} tries left)")
                print(f"Error: {str(e)}")
                if max_tries > 0:
                    return self.execute(obs, short_term_plan, max_tries - 1, verbose)
                else:
                    print("❌ Failed to generate action after all retries")
                    return Action(type=Action.RESUME, code="")
        except Exception as e:
            print(f"❌ Failed to execute action plan: {str(e)}")
            return Action(type=Action.RESUME, code="")
    
    def retry(self, obs, short_term_plan, code_info, max_tries = 3, verbose = False):
        print("\nRetrying failed action...")
        try:
            system_message = self.render_system_message()
            human_message = self.render_human_message(obs, short_term_plan, code_info)
            message = [system_message, human_message]

            try:
                response = self.chain.invoke(message)
                if verbose:
                    print("✅ Successfully generated retry action")
                    print(f"\033[31m****Action Agent (Retry)****\n{response}\033[0m")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write(f"****Action Agent (Retry)****\n{response}\n")
                return Action(type=Action.NEW, code=response["Code"])
            except Exception as e:
                print(f"⚠️ Failed to generate retry action, retrying... ({max_tries} tries left)")
                print(f"Error: {str(e)}")
                if max_tries > 0:
                    return self.retry(obs, short_term_plan, code_info, max_tries - 1, verbose)
                else:
                    print("❌ Failed to generate retry action after all retries")
                    return Action(type=Action.RESUME, code="")
        except Exception as e:
            print(f"❌ Failed to retry action: {str(e)}")
            return Action(type=Action.RESUME, code="")
    
    def redo(self, obs, short_term_plan, critic_info, max_tries = 3, verbose = False):
        print("\nRedoing action with critic feedback...")
        try:
            system_message = self.render_system_message()
            human_message = self.render_human_message(obs, short_term_plan, critic_info=critic_info)
            message = [system_message, human_message]

            try:
                response = self.chain.invoke(message)
                if verbose:
                    print("✅ Successfully generated redo action")
                    print(f"\033[31m****Action Agent (Redo)****\n{response}\033[0m")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write(f"****Action Agent (Redo)****\n{response}\n")
                return Action(type=Action.NEW, code=response["Code"])
            except Exception as e:
                print(f"⚠️ Failed to generate redo action, retrying... ({max_tries} tries left)")
                print(f"Error: {str(e)}")
                if max_tries > 0:
                    return self.redo(obs, short_term_plan, critic_info, max_tries - 1, verbose)
                else:
                    print("❌ Failed to generate redo action after all retries")
                    return Action(type=Action.RESUME, code="")
        except Exception as e:
            print(f"❌ Failed to redo action: {str(e)}")
            return Action(type=Action.RESUME, code="")