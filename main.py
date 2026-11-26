
from pydantic import ValidationError

from app.support_service import SupportService
from app.config import Settings
from app.llm.client import LLMClient
from app.console import run_console

def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    client = LLMClient(settings)
    support_service = SupportService(client)

    run_console(support_service, app_env=settings.app_env, configured_model=settings.model)


if __name__ == "__main__":
    main()
