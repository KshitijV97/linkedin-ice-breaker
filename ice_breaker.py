from langchain.prompts.prompt import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from third_parties.linkedin import scrape_linkedin_profile

information = """

"""


if __name__ == '__main__':
    summary_template = """
        given the linkedin information {information} about a person. I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = ChatOllama(model="llama3")

    chain = summary_prompt_template | llm | StrOutputParser()
    # This creates a processing chain by combining the prompt template and the language model.
    # The | operator indicates that the output of the prompt template will be the input for the language model.

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json")
    res = chain.invoke(input={"information": linkedin_data})

    print(res)