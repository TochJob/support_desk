from dataclasses import dataclass
from time import perf_counter
from app.llm.client import Message

from app.llm.client import LLMClient
from app.llm.errors import LLMClientError

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

class SupportServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupportSummary:
    text:str
    model:str
    elapsed_seconds:float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str | None

class SupportService:
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def summarize(self, user_text:str)-> SupportSummary:
        text = user_text.strip()
        if not text:
            raise SupportServiceError()

        started_at = perf_counter()
        try:
            llm_result = self._llm_client.generate(self._build_messages(text))
        except LLMClientError as error:
            raise SupportServiceError(
                f"Не удалось получить ответ модели: {error}"
            ) from error
        elapsed_seconds = perf_counter() - started_at

        if llm_result.finish_reason == "length":
            raise SupportServiceError(
                "Ответ модели остановлен из-за ограничения длины."
            )
        if llm_result.finish_reason != "stop":
            raise SupportServiceError(
                "Модель не вернула готовое резюме. "
                f"Причина завершения: {llm_result.finish_reason}."
            )
        summary = self._validate_summary(llm_result.text)

        return SupportSummary(
            text= summary,
            model= llm_result.model,
            elapsed_seconds= elapsed_seconds,
            prompt_tokens= llm_result.prompt_tokens,
            completion_tokens= llm_result.completion_tokens,
            total_tokens= llm_result.total_tokens,
            response_id= llm_result.response_id,
        )

    @staticmethod
    def _build_messages(user_text: str) -> list[Message]:
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

    @staticmethod
    def _validate_summary(summary: str | None) -> str:
        if summary is None:
            raise ValueError("Модель не вернула текст")

        cleaned_summary  = summary.strip()
        if not cleaned_summary :
            raise ValueError("Модель вернула пустой ответ")

        return cleaned_summary