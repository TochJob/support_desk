import  os
from os import system
from time import perf_counter

import openai

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-3.1-flash-lite"
SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

def summarize_request(client: OpenAI, user_text:str) -> None:
    text = user_text.strip()
    if not text:
        print("Ошибка: Поле не должно быть пустым")
        return

    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            reasoning_effort="low",
            messages= build_messages(text),
        )
    except openai.AuthenticationError:
        print("Ошибка авторизации: проверьте LLM_API_KEY.")
        return
    except openai.PermissionDeniedError:
        print("Нет доступа к модели: проверьте разрешения API-ключа.")
        return
    except openai.RateLimitError:
        print("Достигнут лимит запросов провайдера. Повторите запрос позднее.")
        return
    except openai.APITimeoutError:
        print("ProxyAPI не успел ответить за отведенное время.")
        return
    except openai.APIConnectionError:
        print("Не удалось соединиться с ProxyAPI. Проверьте сеть.")
        return
    except openai.APIStatusError as error:
        if error.status_code == 402:
            print("Недостаточно средств на балансе ProxyAPI.")
        else:
            print(f"API вернул ошибку со статусом {error.status_code}.")
        if error.request_id:
            print(f"Request ID: {error.request_id}")
        return


    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]
    answer = choice.message.content

    print("\nРезультат:")
    print(answer or "Модель не вернула текстовый ответ")

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")


    print_usage(response.usage)

    print(f"ID ответа: {response.id}")


def print_usage(usage)->None:
    if usage is None:
        print("Провайдер не вернул статистику токенов")
        return

    print(f"Входные токены: {usage.prompt_tokens}")
    print(f"Выходные токены: {usage.completion_tokens}")
    print(f"Всего токенов: {usage.total_tokens}")


def main() -> None:
    load_dotenv()
    token = os.getenv("LLM_API_KEY")
    if not token:
        raise Exception("Please set environment variable LLM_API_KEY")

    client = OpenAI(base_url=BASE_URL, api_key=token, timeout=30, max_retries=3)
    user_text = input('Введите текст сообщения: ')
    summarize_request(client, user_text)


def build_messages(user_text: str) -> list[dict[str, str]]:
    wrapped_input = f"<user_message>\n{user_text}\n</user_message>"
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": wrapped_input,
        },
    ]


if __name__ == "__main__":
    main()
