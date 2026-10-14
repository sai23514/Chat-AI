"""Command-line chat that uses the same documents and provider failover as the web app."""
import uuid

from main import ask_ai, build_messages, friendly_error, save_turn


def main() -> None:
    session_id = str(uuid.uuid4())
    print("Chatbot ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Bye!")
            return
        if question.lower() in {"quit", "exit"}:
            print("Bot: Bye!")
            return
        if not question:
            continue
        try:
            messages, sources = build_messages(question, session_id)
            response, provider = ask_ai(messages, stream=False)
            answer = response.choices[0].message.content or "(No text response)"
            save_turn(session_id, question, answer)
            print(f"Bot [{provider.name}]: {answer}")
            if sources:
                print("Sources:", ", ".join(sources))
            print()
        except Exception as error:
            print(f"Bot: {friendly_error(error)}\n")


if __name__ == "__main__":
    main()
