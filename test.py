from src.rag import load_knowledge_base, split_knowledge

knowledge = load_knowledge_base()

chunks = split_knowledge(knowledge)

for i, chunk in enumerate(chunks):
    print("CHUNK", i + 1)
    print(chunk)
    print("----------------")