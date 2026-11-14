# MailGuard: исправления без нового обучения, 30 октября 2026

Изменения подготовлены отдельно от месячной кампании v82. Веса, зафиксированные
пороги, разметка и исторические отчёты не меняются. Новых optimizer steps нет.
Этот документ описывает инженерные исправления; рост recall или снижение FP на
почте заказчика ими пока не доказаны.

## Что исправлено

| Область | Проблема | Изменение |
|---|---|---|
| Rspamd | `{}` или отсутствующая оценка превращались в ноль | Требуются action и конечные score/required_score; неправильный ответ вызывает ошибку |
| Rspamd | Строковая оценка зависела от локали ОС | Разбор чисел с CultureInfo.InvariantCulture |
| Rspamd | Тело ответа сервера попадало в исключение | Ошибки HTTP не включают содержимое ответа; API не пишет его в лог |
| Rspamd | Не виден прерывающий обработку модуль | В результате сохраняются is_skipped и passthrough_module |
| Решение C# | soft reject мог превратиться в Normal | Если локальная модель не решила SPAM, возвращается Unsure; прежний SPAM сохраняется |
| Кеш | Отмена могла быть проглочена; ошибка записи срывала успешный scan | Отмена распространяется, повреждения перечитываются, ошибки IO не отменяют scan; временные файлы удаляются |
| Кеш | Хеширование копировало всё письмо в дополнительный массив | SHA256 вычисляется последовательно; ключи остаются совместимыми |
| API/CLI | Большое письмо целиком попадало в память | Потоковое чтение с лимитом 20 000 000 байт, включая HTTP без Content-Length |
| API | Сбой Rspamd выглядел как общий HTTP 500 | 502 для недоступного/неправильного backend, 504 для timeout, 413 для размера, 400 для пустого/слишком сложного MIME |
| API | Невозможно отличить живой процесс от загруженной модели | /health остаётся liveness; /ready возвращает 503 без модели |
| MIME C# | Boundary искался внутри текста | Разделителями считаются только строки MIME boundary |
| MIME C# | Текстовое вложение попадало в тело письма | Вложения учитываются отдельно; их содержимое не добавляется к body |
| MIME C# | cp1251/koi8-r не поддерживались, cp1252 читался как Latin-1 | Используется провайдер кодовых страниц .NET |
| MIME C# | Между частями RFC2047 оставался лишний пробел | Удаляется folding whitespace между соседними encoded words |
| MIME C# | Слишком глубокий MIME молча обрезался | Явная ошибка глубины; для сложных HTML regex установлены timeout |
| Модель .mg | Запись могла уничтожить предыдущую модель; неверные размеры/NaN принимались | Атомарная замена, проверка размерности, весов, порогов, счётчиков и конца файла |
| Пороги C# | Перебор занимал O(n²) | Сортировка и проход O(n log n); эквивалентность проверяется исчерпывающим оракулом |
| Python CLI | zip(batch, results) молча терял письма при коротком ответе | Проверяются число результатов, finite scores, язык и согласованность verdict |
| Python CLI | Отказ классификатора мог сорвать отчёт или потеряться | None учитывается как prediction_abstentions, не как HAM |
| Python CLI | Частичный CSV мог выглядеть завершённым | Сначала пишутся .partial; итоговые файлы появляются только после успешного завершения |
| Метрики v80 | Итоговая цель использовала FP / все письма | Основная цель использует FP / HAM; показатели на всех письмах сохранены с отдельными именами |
| v80 | Лексический профиль выполнял лишний MiniLM inference | Лексические кандидаты используют только пять быстрых оценок; прямой CLI не загружает encoder для них |

MIME-исправления относятся к старому C# конвейеру с `.mg`. Они могут изменить его
признаки и решения. Перед заменой реально используемой `.mg`-системы нужна регрессия
на её корпусе с проверкой индивидуальных обнаружений. Python-нормализация и
`parse_eml` современных моделей в этом изменении не меняются.

Общий `attach` v80 по умолчанию сохраняет прежнюю загрузку encoder для research-
инструментов и v81. Оптимизация прямого CLI определяется выбранной policy, а не
одним названием профиля. Семантические кандидаты по-прежнему используют encoder.
Оценки всех семи лексических кандидатов проверяются побитово против прежних формул;
это инженерная проверка, не новая оценка стабильных весов на полном корпусе.

## Проверить у себя

Из корня checkout, .NET 8 SDK и Python 3.12:

```bash
dotnet build MailGuard.sln -c Release
dotnet run --project src/MailGuard.SelfTest -c Release --no-build
python -m pip install -r requirements-v79.txt
python -m unittest discover -s tests/python -p test_runtime_hardening.py -v
python -m unittest discover -s tests/python -p test_rspamd_runtime_check.py -v
python -m unittest discover -s tests/integration -p test_api_runtime.py -v
```

API integration создаёт временную синтетическую модель и HTTP backend. Проверяются
400/413/502/503/504, обычный и chunked request, readiness и отсутствие содержимого
ответа backend в API-логе. Нельзя использовать эту нулевую тестовую модель для почты.

В Docker:

```bash
docker compose --profile benchmark config >/dev/null
docker compose --profile tools run --rm --build mailguard-selftest
docker compose up -d redis rspamd
docker compose exec -T rspamd rspamadm configtest
python tools/check_rspamd_runtime.py
python tools/check_rspamd_runtime.py --smoke-test
```

Для `--smoke-test` отправляются два **синтетических** письма. Только для этих запросов
через Settings отключаются GREYLIST_CHECK/GREYLIST_SAVE: иначе greylisting может
прервать scan раньше GTUBE. Конфигурация сервера не переписывается; статистика и
автообучение сервера могут измениться. Без `--smoke-test` отправляется только ping.
GTUBE проверяет технический путь детектора; эти два сообщения не измеряют качество.
CLI не перезаписывает существующий файл, если задан `--output`.

При ограничительном umask файлы `rspamd/local.d` могут оказаться нечитаемыми для
пользователя Rspamd внутри контейнера. Для поставляемых публичных benchmark-конфигов
проверьте чтение каталога и файлов (обычно 755 и 644). Не применяйте это правило к
каталогам клиентской почты или файлам с секретами.

## Настройки API

| Переменная | По умолчанию | Назначение |
|---|---|---|
| MAILGUARD_MODEL | models/mailguard.mg | Старый C# формат, не model.joblib и не checkpoint.pt |
| MAILGUARD_MAX_MESSAGE_BYTES | 20000000 | Лимит тела запроса, положительное целое |
| RSPAMD_URL | http://localhost:11333/ | Scan endpoint |
| RSPAMD_PASSWORD | отсутствует | Пароль HTTP worker, если требуется |
| RSPAMD_TIMEOUT_SECONDS | 30 | Timeout Rspamd, целое 1–300 |

API/CLI ограничивают буфер HTTP-ответа Rspamd четырьмя миллионами байт.
`/ready` проверяет загрузку модели, а не доступность Rspamd; Rspamd проверяется
отдельно ping или указанным скриптом. Доступный Rspamd может вернуть `soft reject`;
это SMTP-действие временного отказа, а не доказательство SPAM. Интеграция MTA должна
отдельно учитывать исходное действие; C# engine в таком случае возвращает Unsure,
если локальная модель не решила SPAM. Кеш требует новой namespace после изменения
правил/обучения Rspamd: старые результаты не являются свежей проверкой.

## Где ещё возможно улучшать фильтрацию без нейросети

В `rspamd/local.d` намеренно отключены SPF, DKIM, DMARC, RBL и fuzzy: это профиль
воспроизведения публичных benchmark, **не рекомендованная production-конфигурация**.
Контроллер принимает обращения из контейнерной сети без reusable password, порты
Compose привязаны к loopback. Не копируйте этот профиль на публичный почтовый сервер.

Реальный следующий шаг — интеграция с MTA и доверенными IP/HELO/envelope-данными,
проверки аутентификации и согласованности доменов, актуальность DNS/репутационных
источников. Нельзя брать доверенный IP из произвольного заголовка входящего `.eml`
или считать один SPF/DKIM pass достаточным основанием для белого списка. Для
конкретных правил нужны конфигурация и разрешённая локальная разметка заказчика.
Таких данных в этой среде нет, поэтому правила production и пороги не менялись.

Проверки этого изменения выполняются на Linux x86_64. Проверка macOS/Apple Silicon
и Windows отдельно не проводилась. CI дополнен Python/API/Rspamd smoke-проверками.

## Дифференциальная регрессия C# на публичных письмах

Старый Core (`15e5602`) и исправленный Core прогнаны через один и тот же runner на
**3000 публичных письмах SpamAssassin 2003: 2500 HAM + 500 SPAM**. Обе версии разобрали
все письма без ошибок. Разобранное содержимое изменилось у104 писем; признаки — у32,
из них22 SPAM и10 HAM. Это количество изменённых признаков, а не потерянных TP.
Числовой [отчёт с хешами входов и результатов](../reports/runtime-hardening-2026-10-30.json)
не содержит текстов писем. В нём TP/FP-переходы равны null: соответствующего inference
с реальной `.mg` и замороженным контекстом Rspamd не было. Приёмка для внедрения не установлена.

Подходящего проверенного `.mg`-пакета в доступных файлах и проверенных GitHub
артефактах/релизах нет. Исторический public-benchmark workflow не имеет успешных
запусков в просмотренных75 runs; его прежние wrapper-скрипты сейчас отсутствуют.
Сборка Docker benchmark target сама по себе не подтверждает работоспособность этой
старой команды обучения. Новая модель вместо отсутствующей не обучалась.

Для повторения скачайте указанные публичные архивы; подготовщик проверяет закреплённые
SHA256 и записывает только обычные файлы под именами хешей:

```bash
mkdir -p .cache/runtime-archives
curl -fL https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2 -o .cache/runtime-archives/easy_ham.tar.bz2
curl -fL https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2 -o .cache/runtime-archives/spam.tar.bz2
python tools/prepare_runtime_regression_corpus.py --archive-dir .cache/runtime-archives --output .cache/runtime-public
git worktree add --detach ../MailGuard-runtime-baseline 15e5602
dotnet build tools/runtime_regression/RuntimeRegression.csproj -c Release --no-incremental -p:MailGuardCorePath="$(realpath ../MailGuard-runtime-baseline/src/MailGuard.Core/MailGuard.Core.csproj)" -o .cache/runtime-baseline
dotnet build tools/runtime_regression/RuntimeRegression.csproj -c Release --no-incremental -o .cache/runtime-candidate
dotnet .cache/runtime-baseline/RuntimeRegression.dll --input .cache/runtime-public/messages --output .cache/runtime-baseline.jsonl --max-messages 10000
dotnet .cache/runtime-candidate/RuntimeRegression.dll --input .cache/runtime-public/messages --output .cache/runtime-candidate.jsonl --max-messages 10000
python tools/compare_runtime_regression.py --baseline .cache/runtime-baseline.jsonl --candidate .cache/runtime-candidate.jsonl --output .cache/runtime-comparison.json
```

Команды проверены на Linux; в shallow clone сначала потребуется получить commit
`15e5602`. Используйте новые имена output для повторов. Runner сохраняет только хеши
и счётчики, отклоняет дубликаты входных идентичностей и отмечает обрезку выборки.
Опциональный `--model` позволяет сравнить одну `.mg` по одинаковому SHA256, но без
замороженных результатов Rspamd это остаётся проверкой локальной части, не всего
развёрнутого конвейера. Для полноценной приёмки нужны соответствующие веса,
размеченный корпус и одинаковый доверенный SMTP/Rspamd-контекст.

Дополнительный review закрыл отсутствующие обязательные числовые поля кеша,
неправильный тип wire symbols и нормализацию строковых score/cutoff в Python.
