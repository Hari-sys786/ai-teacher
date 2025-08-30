from llm_selector import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    # 1) Create your chain once
    llm   = get_llm("deepseek", stream=True)
    prompt = PromptTemplate.from_template("{query}")
    chain = prompt | llm | StrOutputParser()

    print("🤖 AI Teacher (type 'end', 'stop' or 'exit' to quit)")

    # 2) Loop until user wants to stop
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ("end", "stop", "exit"):
            print("👋 Goodbye!")
            break

        # 3) Invoke the chain with the user's query
        answer = chain.invoke({"query": user_query}).strip()
        print("AI:", answer)

if __name__ == "__main__":
    main()
