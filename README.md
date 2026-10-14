# MailGuard

Фильтр спама с локальной обработкой писем. Последняя законченная модель — экспериментальная v62. На 351 485 письмах без пересечений с обучением и подбором модели: 90,61% распознанного спама, 0 FP/521 HAM. Цель 95% с редкими FP пока не достигнута. [Отчёт проверки 500 000 писем](reports/v62-500k-result.md), [описание набора](docs/v62-500k.md).

## Проверка на своих письмах

Нужны Linux или macOS, Git и работающий Docker; рекомендуется 4 ГБ свободной памяти. Одно письмо — один файл `.eml`. Проверка работает на CPU.

### 1. Скачать код

```bash
git clone --branch mailguard-v48-calibrated --single-branch https://github.com/Kirusha-dk/MailGuard.git MailGuard-check
cd MailGuard-check
mkdir -p model mail/spam mail/ham results
```

Остальные команды выполняйте из `MailGuard-check`.

### 2. Скачать модель

Войдите в GitHub, откройте [запуск v62](https://github.com/Kirusha-dk/MailGuard/actions/runs/38468905929) и в **Artifacts** скачайте **mailguard-v62-evidence**. Распакуйте ZIP в `MailGuard-check/model`. Проверьте:

```bash
ls model/reports/v62-full/model.joblib
```

Артефакт хранится до 11 января 2027 года. [Постоянный пакет для владельца](https://github.com/Kirusha-dk/MailGuard/releases/tag/untagged-c770aa0363895475652e) — отдельный черновик Release. Если используете его `.tar.gz`, распакуйте в `model` и замените путь модели ниже на `/model/v62-full`.

### 3. Подготовить письма

Скопируйте заранее размеченный спам в `mail/spam`, обычную почту в `mail/ham`. Подпапки допустимы, обе папки должны содержать `.eml`. Начните с небольшой выборки, затем проверьте отдельный набор за новый период.

```bash
docker build -f Dockerfile.v62-check -t mailguard-v62-check .
```

Для сборки нужен интернет. Письма в образ не копируются.

### 4. Запустить проверку

```bash
docker run --rm --network none \
  -v "$PWD/model:/model:ro" \
  -v "$PWD/mail:/mail:ro" \
  -v "$PWD/results:/results" \
  mailguard-v62-check \
  --model-dir /model/reports/v62-full \
  --spam /mail/spam --ham /mail/ham \
  --profile high-recall \
  --output /results/check-01
```

Модель и порог не меняются. `high-recall` распознаёт больше спама; `conservative` даёт меньше FP ценой пропусков. Для сравнения замените профиль и папку вывода на `/results/check-02`. При нехватке памяти добавьте `--batch-size 16`.

### 5. Прочитать результат

```bash
cat results/check-01/summary.txt
```

- **Spam recall** — доля обнаруженного спама; **FN** — пропущенный спам.
- **FP** — обычные письма, ошибочно признанные спамом; **FPR** = FP/HAM.
- **failed** — непрочитанные файлы; ненулевое значение означает неполную проверку.
- **Duplicate content / conflicting labels** — повторы и противоречивая разметка. Повторы учтены в итоговых числах.

Рядом сохраняются `summary.json`, `predictions.csv` и `errors.csv`. В CSV нет текстов писем; имена файлов и идентификаторы оставьте у себя. Для обсуждения достаточно `summary.txt` и периода выборки. Действующий фильтр сравнивайте на тех же письмах.

Если Docker не запускается, проверьте, что он включён. Если письма не найдены, проверьте пути и `.eml`. Если папка вывода уже содержит отчёт, используйте новое имя.

Код: `tools/` — обучение и применение моделей; `src/` — исходный C# проект с Rspamd. Качество на почте заказчика не подтверждено. Прежние версии сохранены: [v51](docs/v51-check-legacy.md), [v57](docs/v57-check.md), [v59](docs/v59-check.md), [v60](docs/v60-check.md).

