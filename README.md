# MailGuard

Фильтр спама для почтовых серверов. Проект дополняет Rspamd моделями машинного обучения. Обработка писем — локальная, без внешних AI API.

## Состояние проекта

Разработка и проверка моделей продолжаются в ветке [mailguard-v48-calibrated](https://github.com/Kirusha-dk/MailGuard/tree/mailguard-v48-calibrated).

Результат v51 на 81 688 письмах: **93,16% обнаруженного спама, 80 ложных срабатываний**. Это проверка на публичных наборах данных; результат на почте заказчика ещё предстоит измерить.

## Сборка

Нужен .NET 8 SDK.

```bash
dotnet build MailGuard.sln -c Release
dotnet run --project src/MailGuard.SelfTest -c Release --no-build
```

## Проверка моделей

Обучение и тестирование запускаются через [GitHub Actions](https://github.com/Kirusha-dk/MailGuard/actions). Отчёты, предсказания и модели доступны в артефактах запусков.

Стек: C#, Python, Rspamd, scikit-learn, PyTorch.
