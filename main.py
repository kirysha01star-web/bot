import requests
import json
import time
from typing import Dict, List, Optional


class YandexGPTTester:
    def __init__(self, folder_id: str, api_key: str, model: str = "yandexgpt-lite"):
        """
        Инициализация клиента для YandexGPT

        Args:
            folder_id: Идентификатор каталога Yandex Cloud
            api_key: API-ключ для аутентификации
            model: Модель YandexGPT (yandexgpt-lite, yandexgpt)
        """
        self.folder_id = folder_id
        self.api_key = api_key
        self.model = model
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_text(self,
                      prompt: str,
                      temperature: float = 0.6,
                      max_tokens: int = 2000) -> Dict:
        """
        Генерация текста с помощью YandexGPT

        Args:
            prompt: Текст запроса
            temperature: Креативность (0.0-1.0)
            max_tokens: Максимальное количество токенов

        Returns:
            Словарь с ответом API
        """

        data = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты полезный AI-ассистент от Яндекс"
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": response.status_code if 'response' in locals() else None}

    def test_basic_functionality(self) -> Dict:
        """
        Базовый тест функциональности YandexGPT
        """
        tests = [
            {
                "name": "Простой вопрос",
                "prompt": "Привет! Как дела?"
            },
            {
                "name": "Фактический вопрос",
                "prompt": "Сколько планет в Солнечной системе?"
            },
            {
                "name": "Креативная задача",
                "prompt": "Напиши короткое стихотворение о Python"
            }
        ]

        results = {}

        for test in tests:
            print(f"\nТест: {test['name']}")
            print(f"Запрос: {test['prompt']}")

            result = self.generate_text(test['prompt'])

            if "result" in result:
                answer = result["result"]["alternatives"][0]["message"]["text"]
                print(f"Ответ: {answer[:100]}...")
                results[test['name']] = {
                    "status": "success",
                    "response_length": len(answer),
                    "truncated_response": answer[:200] + "..." if len(answer) > 200 else answer
                }
            else:
                print(f"Ошибка: {result.get('error', 'Unknown error')}")
                results[test['name']] = {
                    "status": "error",
                    "error": result.get('error', 'Unknown error')
                }

            time.sleep(1)  # Задержка между запросами

        return results

    def test_performance(self, prompt: str = "Расскажи о преимуществах искусственного интеллекта",
                         num_requests: int = 3) -> List[Dict]:
        """
        Тестирование производительности

        Args:
            prompt: Текст запроса
            num_requests: Количество запросов для теста
        """
        results = []

        for i in range(num_requests):
            print(f"\nЗапрос {i + 1}/{num_requests}...")

            start_time = time.time()
            response = self.generate_text(prompt)
            end_time = time.time()

            if "result" in response:
                response_time = end_time - start_time
                token_count = len(response["result"]["alternatives"][0]["message"]["text"])

                result = {
                    "request_number": i + 1,
                    "status": "success",
                    "response_time": round(response_time, 2),
                    "token_count": token_count,
                    "tokens_per_second": round(token_count / response_time, 2) if response_time > 0 else 0
                }
                results.append(result)

                print(f"Время ответа: {response_time:.2f} сек")
                print(f"Токенов: {token_count}")
            else:
                print(f"Ошибка: {response.get('error', 'Unknown error')}")
                results.append({
                    "request_number": i + 1,
                    "status": "error",
                    "error": response.get('error', 'Unknown error')
                })

            time.sleep(1)

        return results

    def run_comprehensive_test(self) -> Dict:
        """
        Комплексный тест работы YandexGPT
        """
        print("=" * 50)
        print("НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ YANDEXGPT")
        print("=" * 50)

        summary = {
            "basic_tests": {},
            "performance_tests": [],
            "overall_status": "pending"
        }

        # Тест базовой функциональности
        print("\n1. ТЕСТ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ")
        print("-" * 30)
        summary["basic_tests"] = self.test_basic_functionality()

        # Тест производительности
        print("\n\n2. ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("-" * 30)
        summary["performance_tests"] = self.test_performance()

        # Оценка результатов
        basic_success = all(
            test.get("status") == "success"
            for test in summary["basic_tests"].values()
        )

        performance_success = all(
            test.get("status") == "success"
            for test in summary["performance_tests"]
        )

        summary["overall_status"] = "success" if basic_success and performance_success else "partial_failure"

        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print("=" * 50)
        print(f"Базовые тесты: {'Пройдены' if basic_success else 'Есть ошибки'}")
        print(f"Тесты производительности: {'Пройдены' if performance_success else 'Есть ошибки'}")
        print(f"Общий статус: {summary['overall_status']}")

        return summary


# Пример использования
def main():
    # Замените эти значения на свои
    FOLDER_ID = "b1go5q51mq9uqbvnl79i"
    API_KEY = "AQVNkxexRxUZi93gsg5Gpae24L8GhK.LX25300na"

    # Инициализация тестера
    tester = YandexGPTTester(
        folder_id=FOLDER_ID,
        api_key=API_KEY,
        model="yandexgpt-lite"  # Можно изменить на "yandexgpt" для полной версии
    )

    # Запуск комплексного теста
    results = tester.run_comprehensive_test()

    # Сохранение результатов в файл
    with open("yandexgpt_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сохранены в файл: yandexgpt_test_results.json")


# Простой тест без сохранения результатов
def quick_test():
    FOLDER_ID = "b1go5q51mq9uqbvnl79i"
    API_KEY = "AQVNkxexRxUZi93gsg5Gpae24L8GhK.LX25300na"

    tester = YandexGPTTester(FOLDER_ID, API_KEY)

    # Одиночный запрос
    prompt = "Напиши код на Python для сложения двух чисел"
    response = tester.generate_text(prompt)

    if "result" in response:
        answer = response["result"]["alternatives"][0]["message"]["text"]
        print("Ответ YandexGPT:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
    else:
        print(f"Ошибка: {response}")


if __name__ == "__main__":
    # Выберите нужный способ тестирования
    # quick_test()  # Быстрый тест
    main()  # Полное тестирование